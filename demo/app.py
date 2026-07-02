"""
Learnity RecSys -- интерактивная демонстрация (Streamlit).

Запуск:
    cd learnity-recsys
    streamlit run demo/app.py

Все вычисления происходят in-memory; запущенные сервисы и БД не требуются.
"""

from __future__ import annotations

import math
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.profile.bkt import smooth_update, compute_confidence
from services.retrieval.thompson import ThompsonModel
from services.retrieval.diagnostic_cat import (
    CATState,
    select_diagnostic_kc,
    select_diagnostic_task,
    update_cat_state,
)
from services.macro.diagnostics import diagnose, Diagnosis
from services.macro.student_profile import MacroStudentProfile
from services.macro.estimators import estimate_tasks_to_mastery, estimate_stall_risk, estimate_regression_risk
from services.retrieval.selector import (
    filter_tasks_by_irt,
    compute_p_correct,
    compute_zpd_target_difficulty,
    select_task,
)
from services.clustering.cluster import _select_n_clusters
from services.graph.kc_data import (
    KC_GRAPH, KC_NAMES, KC_INTRO_GRADE, KC_SUBJECTS,
    ALL_KC_IDS, SUBJECT_RU, EDGES, EDGE_STRENGTHS,
)
from shared.config import retrieval as _rcfg, bkt as _bcfg

CONTEXT_DIM = _rcfg.CONTEXT_DIM

st.set_page_config(page_title="Learnity RecSys Demo", page_icon=":books:", layout="wide")

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* hide default streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* metric cards */
[data-testid="stMetric"] {
    background: rgba(34, 197, 94, 0.06);
    border: 1px solid rgba(34, 197, 94, 0.18);
    border-radius: 10px;
    padding: 12px 16px;
}

/* reduce top padding */
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly global template
# ---------------------------------------------------------------------------
_plotly_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family="sans-serif", color="#a1a1aa", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,9,11,0.8)",
        colorway=["#22c55e", "#3b82f6", "#ef4444", "#a855f7",
                   "#f97316", "#06b6d4", "#f43f5e", "#84cc16"],
        xaxis=dict(gridcolor="rgba(63,63,70,0.5)", zerolinecolor="rgba(63,63,70,0.7)"),
        yaxis=dict(gridcolor="rgba(63,63,70,0.5)", zerolinecolor="rgba(63,63,70,0.7)"),
        margin=dict(l=40, r=20, t=40, b=40),
        hoverlabel=dict(
            bgcolor="rgba(24,24,27,0.97)",
            bordercolor="rgba(34,197,94,0.3)",
            font=dict(size=12, color="#e4e4e7"),
        ),
    ),
)

import plotly.io as pio
pio.templates["learnity"] = _plotly_template
pio.templates.default = "plotly_dark+learnity"

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(34,197,94,0.08) 0%, rgba(9,9,11,0.9) 100%);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
">
    <h1 style="margin:0 0 6px 0; font-size:2rem; font-weight:700;
        color: #22c55e;">
        Learnity RecSys
    </h1>
    <p style="margin:0 0 12px 0; font-size:1.05rem; color:#a1a1aa;">
        Адаптивная рекомендательная система для персонализированного обучения математике
    </p>
    <div style="display:flex; gap:24px; flex-wrap:wrap; font-size:13px; color:#71717a;">
        <span><b style="color:#22c55e;">142</b> Knowledge Components</span>
        <span><b style="color:#22c55e;">253</b> связи в графе</span>
        <span><b style="color:#22c55e;">IRT + EMA + Thompson</b> пайплайн</span>
        <span><b style="color:#22c55e;">5-11</b> класс</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================================================================
# Граф знаний (из services/graph/kc_data.py — реальный граф проекта)
# ===================================================================

ALL_KCS = sorted(ALL_KC_IDS)

STUDENT_TYPES = {
    "Средний": {"growth": 0.07, "p_slip": 0.08, "p_guess": 0.08, "mod": 0.0},
    "Быстрый": {"growth": 0.12, "p_slip": 0.05, "p_guess": 0.06, "mod": 0.08},
    "Медленный": {"growth": 0.04, "p_slip": 0.15, "p_guess": 0.10, "mod": -0.10},
    "Продвинутый": {"growth": 0.09, "p_slip": 0.04, "p_guess": 0.05, "mod": 0.18},
}

DIAGNOSIS_RU = {
    "prereq_gap": ("Пробел в пререквизитах", "Слабый пререквизит тормозит освоение"),
    "content_gap": ("Недостаток контента", "Мало заданий подходящей сложности"),
    "uncertain_estimate": ("Неуверенная оценка", "Недостаточно данных для оценки"),
    "regression": ("Регрессия", "Ученик забывает ранее освоенный материал"),
    "on_track": ("Нормальный прогресс", "Ученик осваивает тему в нормальном темпе"),
}


# ===================================================================
# Вспомогательные функции
# ===================================================================

def init_true_mastery(grade: int, mod: float) -> dict[str, float]:
    mastery = {}
    for kc, intro in KC_INTRO_GRADE.items():
        years = grade - intro
        if years < 0:
            base = 0.03
        elif years == 0:
            base = 0.30
        elif years == 1:
            base = 0.50
        elif years == 2:
            base = 0.65
        else:
            base = min(0.95, 0.70 + 0.04 * years)
        mastery[kc] = max(0.02, min(0.98, base + mod))
    return mastery


def grade_kcs(grade: int) -> list[str]:
    return sorted(kc for kc, intro in KC_INTRO_GRADE.items() if intro <= grade)


def available_targets(grade: int) -> list[str]:
    return sorted(kc for kc, intro in KC_INTRO_GRADE.items() if intro <= grade + 1)


def make_task_pool(kc_ids: list[str], n_per_kc: int = 7) -> list[dict]:
    tasks = []
    rng = random.Random(0)
    for kc in kc_ids:
        for i in range(n_per_kc):
            diff = -2.0 + 4.0 * i / max(1, n_per_kc - 1) + rng.gauss(0, 0.12)
            tasks.append({
                "task_id": f"{kc}_t{i}",
                "kc_id": kc,
                "parts": [{"irt_difficulty": round(diff, 3)}],
            })
    return tasks


TASK_POOL = make_task_pool(ALL_KCS)


def diff_label(d: float) -> str:
    if d < -1.0:
        return "легкое"
    if d < 0.0:
        return "ниже среднего"
    if d < 1.0:
        return "среднее"
    return "сложное"


def simulate_answer(true_mastery: float, irt_difficulty: float,
                    p_slip: float, p_guess: float, rng: random.Random) -> float:
    m = max(0.001, min(0.999, true_mastery))
    theta = math.log(m / (1.0 - m))
    p_irt = 1.0 / (1.0 + math.exp(-(theta - irt_difficulty)))
    p_correct = p_guess + (1.0 - p_guess - p_slip) * p_irt
    r = rng.random()
    if r < p_correct:
        return 1.0
    if r < p_correct + 0.08 * (1.0 - p_correct):
        return 0.5
    return 0.0


def build_learning_path(target_kc: str, mastery: dict[str, float],
                        threshold: float = 0.75) -> list[str]:
    def _collect(kc: str, visited: set) -> list[str]:
        if kc in visited:
            return []
        visited.add(kc)
        result = []
        for prereq in KC_GRAPH.get(kc, []):
            result.extend(_collect(prereq, visited))
        if mastery.get(kc, 0.0) < threshold:
            result.append(kc)
        return result

    path = _collect(target_kc, set())
    seen: set[str] = set()
    return [kc for kc in path if not (kc in seen or seen.add(kc))]  # type: ignore[func-returns-value]


def compute_micro_summary(history: list[dict], kc_id: str,
                          mastery_current: float, window: int = 10) -> dict:
    kc_hist = [h for h in history if h["kc_id"] == kc_id][-window:]
    if not kc_hist:
        return {"velocity": 0.0, "frustration_count": 0, "avg_score": 0.0,
                "irt_residual": 0.0, "sample_size": 0}
    scores = [h["score"] for h in kc_hist]
    deltas = [h.get("mastery_delta", 0.0) for h in kc_hist]
    mid = len(deltas) // 2
    if mid > 0:
        velocity = sum(deltas[:mid]) / mid - sum(deltas[mid:]) / (len(deltas) - mid)
    elif deltas:
        velocity = sum(deltas) / len(deltas)
    else:
        velocity = 0.0
    frust = 0
    for s in reversed(scores):
        if s < 0.5:
            frust += 1
        else:
            break
    residuals = [abs(compute_p_correct(mastery_current, h["irt_diff"]) - h["score"])
                 for h in kc_hist if h.get("irt_diff") is not None]
    return {
        "velocity": round(velocity, 4),
        "frustration_count": frust,
        "avg_score": round(sum(scores) / len(scores), 4),
        "irt_residual": round(sum(residuals) / len(residuals), 4) if residuals else 0.0,
        "sample_size": len(scores),
    }


def score_label(s: float) -> str:
    if s >= 0.9:
        return "верно"
    if s >= 0.4:
        return "частично"
    return "неверно"


# ===================================================================
# Вкладки
# ===================================================================

tab_full, tab_micro, tab_macro, tab_mastery = st.tabs([
    "Полная симуляция",
    "Micro-уровень",
    "Macro-уровень",
    "Mastery и Confidence",
])


# ===================================================================
# TAB 1 -- Полная симуляция системы
# ===================================================================
with tab_full:
    st.header("Полная симуляция адаптивного обучения")
    st.markdown(
        "Пошаговая демонстрация полного цикла: диагностика нового ученика, "
        "кластеризация, построение плана и прохождение с взаимодействием micro/macro."
    )

    col_cfg, col_viz = st.columns([1, 3])

    with col_cfg:
        st.subheader("Параметры ученика")
        grade = st.slider("Класс", 1, 11, 8, key="full_grade")
        student_type = st.selectbox("Тип ученика", list(STUDENT_TYPES.keys()), key="full_type")
        st_params = STUDENT_TYPES[student_type]

        targets = available_targets(grade)
        target_kc = st.selectbox(
            "Целевая тема",
            targets,
            format_func=lambda x: f"{KC_NAMES[x]} (класс {KC_INTRO_GRADE[x]})",
            key="full_target",
        )
        mastery_threshold = st.slider("Порог mastery", 0.60, 0.95, 0.75, 0.05, key="full_thr")
        cat_budget = st.slider("Бюджет CAT (заданий)", 8, 40, 20, key="full_cat_budget")
        run_full = st.button("Запустить симуляцию", type="primary", key="full_run")

    with col_viz:
        if not run_full:
            st.info("Задайте параметры и нажмите **Запустить симуляцию**.")
        else:
            rng = random.Random(42)
            true_m = init_true_mastery(grade, st_params["mod"])
            p_slip = st_params["p_slip"]
            p_guess = st_params["p_guess"]
            growth = st_params["growth"]
            sim_kcs = grade_kcs(grade)

            # ============================================================
            # ФАЗА 1: Диагностический CAT
            # ============================================================
            st.subheader("Фаза 1: Диагностический тест (CAT)")
            st.markdown(
                f"Система не знает уровень ученика ({len(sim_kcs)} тем для {grade} класса). "
                f"Она проводит до **{cat_budget}** адаптивных заданий, "
                "выбирая те, где информация Фишера максимальна (P(correct) ~ 0.5). "
                "После каждого ответа информация распространяется по графу знаний "
                "на пререквизиты и зависимые темы."
            )

            cat_priors = {kc: 0.50 for kc in sim_kcs}
            cat_state = CATState.from_mastery(cat_priors)
            cat_log: list[dict] = []

            # Обратный граф: kc -> list of dependents
            _dependents: dict[str, list[str]] = {kc: [] for kc in sim_kcs}
            for kc in sim_kcs:
                for prereq in KC_GRAPH.get(kc, []):
                    if prereq in _dependents:
                        _dependents[prereq].append(kc)

            def _propagate_cat(state: CATState, tested_kc: str, score: float) -> None:
                """Транзитивная пропагация по графу после ответа."""
                tested_theta = state.kc_theta[tested_kc]
                if score >= 0.5:
                    # Ученик справился -> пререквизиты скорее всего освоены.
                    # BFS назад по графу: theta пререквизита >= theta протестированного * decay
                    queue = [(p, 0.85) for p in KC_GRAPH.get(tested_kc, []) if p in state.kc_theta]
                    visited = {tested_kc}
                    while queue:
                        kc, decay = queue.pop(0)
                        if kc in visited:
                            continue
                        visited.add(kc)
                        floor = tested_theta * decay
                        if state.kc_theta[kc] < floor:
                            state.kc_theta[kc] = floor
                        for p in KC_GRAPH.get(kc, []):
                            if p in state.kc_theta and p not in visited:
                                queue.append((p, decay * 0.8))
                else:
                    # Ученик не справился -> зависимые темы, вероятно, тоже слабы.
                    queue = [(d, 0.8) for d in _dependents.get(tested_kc, []) if d in state.kc_theta]
                    visited = {tested_kc}
                    while queue:
                        kc, decay = queue.pop(0)
                        if kc in visited:
                            continue
                        visited.add(kc)
                        ceiling = tested_theta * decay
                        if state.kc_theta[kc] > ceiling:
                            state.kc_theta[kc] = ceiling
                        for d in _dependents.get(kc, []):
                            if d in state.kc_theta and d not in visited:
                                queue.append((d, decay * 0.8))

            # Ранжируем KCs: сначала хаб-ноды (макс. связей), потом наименее протестированные
            _kc_connectivity = {}
            for kc in sim_kcs:
                n_conn = len(KC_GRAPH.get(kc, [])) + len(_dependents.get(kc, []))
                _kc_connectivity[kc] = n_conn

            def _select_cat_kc(state: CATState) -> str | None:
                candidates = [kc for kc in sim_kcs if kc in state.kc_theta]
                if not candidates:
                    return None
                return min(candidates, key=lambda kc: (
                    state.kc_n.get(kc, 0),
                    -_kc_connectivity.get(kc, 0),
                    abs(state.kc_theta.get(kc, 0.0)),
                ))

            tasks_done_cat = 0
            while tasks_done_cat < cat_budget:
                kc = _select_cat_kc(cat_state)
                if kc is None:
                    break
                kc_tasks = [t for t in TASK_POOL if t["kc_id"] == kc]
                task = select_diagnostic_task(kc_tasks, cat_state.kc_theta[kc])
                if task is None:
                    break
                irt_diff = task["parts"][0]["irt_difficulty"]
                score = simulate_answer(true_m[kc], irt_diff, p_slip, p_guess, rng)
                mastery_before = 1.0 / (1.0 + math.exp(-cat_state.kc_theta[kc]))
                # Усиленный CAT-update: lr=2/(1+n) вместо 1/(1+n) для более агрессивной калибровки
                n_prev = cat_state.kc_n.get(kc, 0)
                p_c = 1.0 / (1.0 + math.exp(-(cat_state.kc_theta[kc] - irt_diff)))
                cat_lr = 2.0 / (1.0 + n_prev)
                cat_state.kc_theta[kc] += cat_lr * (score - p_c)
                cat_state.kc_n[kc] = n_prev + 1
                cat_state.tasks_used += 1
                _propagate_cat(cat_state, kc, score)
                mastery_after = 1.0 / (1.0 + math.exp(-cat_state.kc_theta[kc]))
                cat_log.append({
                    "N": len(cat_log) + 1,
                    "Тема": KC_NAMES[kc],
                    "Сложность": diff_label(irt_diff),
                    "Ответ": score_label(score),
                    "Mastery до": f"{mastery_before:.2f}",
                    "Mastery после": f"{mastery_after:.2f}",
                    "Истинный": f"{true_m[kc]:.2f}",
                })
                tasks_done_cat += 1

            vis_m = cat_state.to_mastery()

            # Итеративная таблица CAT
            st.markdown("**Пошаговая история CAT-диагностики:**")
            st.dataframe(pd.DataFrame(cat_log), use_container_width=True, hide_index=True)

            # Сравнение: до / после / истинный
            st.markdown("**Результат калибровки:**")
            comp_rows = []
            for kc in sim_kcs:
                comp_rows.append({
                    "Тема": KC_NAMES[kc],
                    "Prior": 0.50,
                    "После CAT": round(vis_m[kc], 2),
                    "Истинный": round(true_m[kc], 2),
                    "Ошибка": round(abs(vis_m[kc] - true_m[kc]), 2),
                })
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

            mae_prior = float(np.mean([abs(0.50 - true_m[kc]) for kc in sim_kcs]))
            mae_cat = float(np.mean([abs(vis_m[kc] - true_m[kc]) for kc in sim_kcs]))
            c1, c2, c3 = st.columns(3)
            c1.metric("Заданий в CAT", len(cat_log))
            c2.metric("MAE до CAT", f"{mae_prior:.3f}")
            c3.metric("MAE после CAT", f"{mae_cat:.3f}",
                       delta=f"{mae_cat - mae_prior:+.3f}", delta_color="inverse")

            st.divider()

            # ============================================================
            # ФАЗА 2: Кластеризация
            # ============================================================
            st.subheader("Фаза 2: Определение кластера ученика")
            st.markdown(
                "Система сравнивает mastery-профиль ученика с другими "
                "и определяет его группу (GMM + BIC)."
            )

            rng_np = np.random.RandomState(42)

            group_centers = [-0.15, -0.05, 0.05, 0.15]
            peers_per_group = 20
            peer_rows = []
            for center in group_centers:
                for _ in range(peers_per_group):
                    mod_i = center + rng_np.randn() * 0.04
                    peer_m = init_true_mastery(grade, mod_i)
                    base = np.array([peer_m[kc] for kc in sim_kcs], dtype=np.float32)
                    blend = 0.3 + rng_np.rand() * 0.4
                    row = base * (1 - blend) + base.mean() * blend
                    row += rng_np.randn(len(sim_kcs)).astype(np.float32) * 0.08
                    peer_rows.append(np.clip(row, 0.02, 0.98))

            peer_data = np.array(peer_rows, dtype=np.float32)
            student_vec = np.array([vis_m[kc] for kc in sim_kcs], dtype=np.float32)
            full_matrix = np.vstack([peer_data, student_vec.reshape(1, -1)])
            gmm, best_k = _select_n_clusters(full_matrix, max_k=8)
            all_labels = gmm.predict(full_matrix)
            student_cluster = int(all_labels[-1])

            pca = PCA(n_components=2)
            coords = pca.fit_transform(full_matrix)

            peer_avg_mastery = [float(peer_data[i].mean()) for i in range(len(peer_data))]

            fig_cl = go.Figure()
            cluster_ids_sorted = sorted(set(all_labels[:-1]))
            palette = px.colors.qualitative.Set2
            for ci in cluster_ids_sorted:
                mask = [j for j in range(len(peer_data)) if all_labels[j] == ci]
                avg_m = np.mean([peer_avg_mastery[j] for j in mask])
                fig_cl.add_trace(go.Scatter(
                    x=coords[mask, 0], y=coords[mask, 1],
                    mode="markers",
                    marker=dict(size=9, color=palette[ci % len(palette)], opacity=0.75),
                    name=f"Кластер {ci} (avg {avg_m:.2f})",
                    text=[f"avg mastery: {peer_avg_mastery[j]:.2f}" for j in mask],
                    hovertemplate="%{text}<extra></extra>",
                ))
            fig_cl.add_trace(go.Scatter(
                x=[coords[-1, 0]], y=[coords[-1, 1]],
                mode="markers",
                marker=dict(size=14, color="#ef4444",
                            line=dict(width=2, color="white")),
                name="Наш ученик",
                hovertemplate="Наш ученик<extra></extra>",
            ))
            fig_cl.update_layout(
                height=400, margin=dict(t=10, b=10),
                xaxis_title="PC1", yaxis_title="PC2",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cl, use_container_width=True)

            cluster_size = int((all_labels == student_cluster).sum())
            cluster_centroid = gmm.means_[student_cluster]
            cluster_avg_m = float(cluster_centroid.mean())
            st.success(
                f"Найдено **{best_k}** кластеров (GMM + BIC). "
                f"Ученик отнесён к **кластеру {student_cluster}** "
                f"({cluster_size} уч., средний mastery кластера: {cluster_avg_m:.2f})."
            )

            st.markdown("**Как кластер влияет на подбор заданий:**")
            cluster_members = [j for j in range(len(peer_data)) if all_labels[j] == student_cluster]
            if cluster_members:
                cluster_matrix = peer_data[cluster_members]
            else:
                cluster_matrix = peer_data
            cluster_mean_per_kc = cluster_matrix.mean(axis=0)

            rng_eff = np.random.RandomState(student_cluster)
            n_sim_tasks = 12
            sim_task_kcs = rng_eff.choice(len(sim_kcs), size=n_sim_tasks, replace=False)
            effect_rows = []
            for idx in sim_task_kcs:
                kc = sim_kcs[idx]
                cluster_avg_kc = float(cluster_mean_per_kc[idx])
                student_m_kc = vis_m.get(kc, 0.5)
                sim_reward = max(0.0, min(1.0, cluster_avg_kc * 0.6 + 0.2 + rng_eff.randn() * 0.08))
                sim_count = int(rng_eff.randint(2, 20))
                effect_rows.append({
                    "Тема": KC_NAMES[kc],
                    "Mastery ученика": round(student_m_kc, 2),
                    "Avg mastery кластера": round(cluster_avg_kc, 2),
                    "Avg reward кластера": round(sim_reward, 2),
                    "Взаимодействий": sim_count,
                    "Эффект": "cluster explore" if sim_count < 5 else "Thompson prior",
                })
            st.dataframe(pd.DataFrame(effect_rows), use_container_width=True, hide_index=True)
            st.caption(
                "**cluster explore** — задание мало тестировалось в кластере, система "
                "приоритизирует его для сбора статистики (20% шанс в режиме build). "
                "**Thompson prior** — средняя награда кластера используется как "
                "начальное приближение для бандита (x[7] контекстного вектора)."
            )

            st.divider()

            # ============================================================
            # ФАЗА 3: Построение плана
            # ============================================================
            st.subheader("Фаза 3: Построение плана обучения")

            plan = build_learning_path(target_kc, vis_m, mastery_threshold)

            if not plan:
                st.success(
                    f"Ученик уже освоил **{KC_NAMES[target_kc]}** "
                    f"и все пререквизиты (mastery >= {mastery_threshold})!"
                )
            else:
                st.markdown(
                    f"**Цель:** {KC_NAMES[target_kc]} "
                    f"(mastery {vis_m.get(target_kc, 0):.2f} -> {mastery_threshold})"
                )

                # -- Полный граф знаний с выделенным маршрутом (cytoscape.js) --
                import json as _json
                import streamlit.components.v1 as components

                def _collect_subgraph(target: str) -> set[str]:
                    nodes: set[str] = set()
                    def _walk(kc: str) -> None:
                        if kc in nodes:
                            return
                        nodes.add(kc)
                        for p in KC_GRAPH.get(kc, []):
                            _walk(p)
                    _walk(target)
                    return nodes

                subgraph_kcs = _collect_subgraph(target_kc)
                plan_set = set(plan)

                cy_nodes = []
                cy_edges = []

                for kc in sim_kcs:
                    m_val = vis_m.get(kc, 0)
                    in_subgraph = kc in subgraph_kcs
                    if kc == target_kc:
                        node_type = "target"
                    elif kc in plan_set:
                        node_type = "plan"
                    elif in_subgraph and m_val >= mastery_threshold:
                        node_type = "mastered"
                    elif in_subgraph:
                        node_type = "subgraph"
                    else:
                        node_type = "outside"
                    plan_order = plan.index(kc) + 1 if kc in plan_set else 0
                    subj = KC_SUBJECTS.get(kc, "")
                    subj_ru = SUBJECT_RU.get(subj, subj)
                    cy_nodes.append({
                        "data": {
                            "id": kc,
                            "label": KC_NAMES[kc],
                            "mastery": round(m_val, 2),
                            "mastery_pct": f"{m_val:.0%}",
                            "node_type": node_type,
                            "plan_order": plan_order,
                            "in_subgraph": in_subgraph,
                            "subject": subj_ru,
                            "grade": KC_INTRO_GRADE.get(kc, 0),
                        }
                    })

                sim_kcs_set = set(sim_kcs)
                for kc in sim_kcs:
                    for prereq in KC_GRAPH.get(kc, []):
                        if prereq not in sim_kcs_set:
                            continue
                        is_plan_edge = prereq in plan_set and kc in plan_set
                        is_subgraph_edge = prereq in subgraph_kcs and kc in subgraph_kcs
                        if is_plan_edge:
                            etype = "plan"
                        elif is_subgraph_edge:
                            etype = "subgraph"
                        else:
                            etype = "outside"
                        strength = EDGE_STRENGTHS.get((prereq, kc), 0.5)
                        cy_edges.append({
                            "data": {
                                "source": prereq,
                                "target": kc,
                                "edge_type": etype,
                                "strength": round(strength, 2),
                            }
                        })

                cy_elements = _json.dumps(cy_nodes + cy_edges)
                plan_ids = _json.dumps(plan)

                cy_html = f"""
<div style="position:relative;">
<div id="cy-graph" style="width:100%;height:520px;border-radius:14px;
     background:#0e1117;position:relative;overflow:hidden;
     border:1px solid rgba(255,255,255,0.06);">
</div>
<div id="cy-tooltip" style="position:absolute;display:none;padding:10px 14px;
     background:rgba(14,17,23,0.96);color:#e2e8f0;border-radius:10px;
     font-size:13px;pointer-events:none;z-index:100;
     border:1px solid rgba(99,102,241,0.35);
     box-shadow:0 8px 30px rgba(0,0,0,0.5);
     backdrop-filter:blur(12px);max-width:220px;"></div>
</div>

<div style="display:flex;gap:18px;margin-top:8px;padding:6px 2px;
     font-size:12.5px;color:#94a3b8;flex-wrap:wrap;align-items:center;">
  <span style="display:flex;align-items:center;gap:5px;">
    <span style="width:16px;height:16px;border-radius:4px;
      background:linear-gradient(135deg,#818cf8,#6366f1);
      box-shadow:0 0 10px rgba(99,102,241,0.5);"></span>
    Целевая тема</span>
  <span style="display:flex;align-items:center;gap:5px;">
    <span style="width:16px;height:16px;border-radius:4px;
      border:2px solid #f59e0b;background:rgba(245,158,11,0.15);"></span>
    Маршрут обучения</span>
  <span style="display:flex;align-items:center;gap:5px;">
    <span style="width:16px;height:16px;border-radius:4px;
      background:#10b981;"></span>
    Освоено</span>
  <span style="display:flex;align-items:center;gap:5px;">
    <span style="width:16px;height:16px;border-radius:4px;
      background:#334155;border:1px solid #475569;"></span>
    Пререквизит (в подграфе)</span>
  <span style="display:flex;align-items:center;gap:5px;">
    <span style="width:16px;height:16px;border-radius:4px;
      background:#1e293b;border:1px solid #334155;"></span>
    Вне подграфа</span>
</div>

<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {{

  function masteryColor(m) {{
    if (m >= 0.75) return '#10b981';
    if (m >= 0.50) return '#3b82f6';
    if (m >= 0.30) return '#f59e0b';
    return '#ef4444';
  }}

  var elements = {cy_elements};
  elements.forEach(function(el) {{
    if (el.data && el.data.mastery !== undefined) {{
      el.data._bg = masteryColor(el.data.mastery);
    }}
  }});

  var cy = cytoscape({{
    container: document.getElementById('cy-graph'),
    elements: elements,
    style: [
      /* --- outside nodes: dim --- */
      {{
        selector: 'node[node_type="outside"]',
        style: {{
          'label': 'data(label)',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '9px',
          'font-family': "'Inter',system-ui,sans-serif",
          'color': '#64748b',
          'text-wrap': 'wrap',
          'text-max-width': '80px',
          'background-color': '#1e293b',
          'width': 52, 'height': 52,
          'shape': 'round-rectangle',
          'border-width': 1,
          'border-color': '#334155',
          'opacity': 0.5,
        }}
      }},
      /* --- subgraph nodes (mastered prereqs) --- */
      {{
        selector: 'node[node_type="mastered"]',
        style: {{
          'label': 'data(label)',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '10.5px',
          'font-family': "'Inter',system-ui,sans-serif",
          'font-weight': '500',
          'color': '#f0fdf4',
          'text-wrap': 'wrap',
          'text-max-width': '85px',
          'background-color': '#10b981',
          'width': 66, 'height': 66,
          'shape': 'round-rectangle',
          'border-width': 2,
          'border-color': '#34d399',
          'text-outline-width': 0,
        }}
      }},
      /* --- subgraph nodes (not mastered, not in plan) --- */
      {{
        selector: 'node[node_type="subgraph"]',
        style: {{
          'label': 'data(label)',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '10.5px',
          'font-family': "'Inter',system-ui,sans-serif",
          'font-weight': '500',
          'color': '#e2e8f0',
          'text-wrap': 'wrap',
          'text-max-width': '85px',
          'background-color': '#334155',
          'width': 66, 'height': 66,
          'shape': 'round-rectangle',
          'border-width': 2,
          'border-color': '#64748b',
        }}
      }},
      /* --- plan nodes: bright amber --- */
      {{
        selector: 'node[node_type="plan"]',
        style: {{
          'label': 'data(label)',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '11px',
          'font-family': "'Inter',system-ui,sans-serif",
          'font-weight': '600',
          'color': '#fffbeb',
          'text-wrap': 'wrap',
          'text-max-width': '90px',
          'background-color': 'data(_bg)',
          'width': 74, 'height': 74,
          'shape': 'round-rectangle',
          'border-width': 3,
          'border-color': '#f59e0b',
          'shadow-blur': 12,
          'shadow-color': 'rgba(245,158,11,0.35)',
          'shadow-opacity': 1,
        }}
      }},
      /* --- target node: indigo glow --- */
      {{
        selector: 'node[node_type="target"]',
        style: {{
          'label': 'data(label)',
          'text-valign': 'center',
          'text-halign': 'center',
          'font-size': '12px',
          'font-family': "'Inter',system-ui,sans-serif",
          'font-weight': '700',
          'color': '#eef2ff',
          'text-wrap': 'wrap',
          'text-max-width': '95px',
          'background-color': '#6366f1',
          'width': 88, 'height': 88,
          'shape': 'round-rectangle',
          'border-width': 3,
          'border-color': '#818cf8',
          'shadow-blur': 24,
          'shadow-color': 'rgba(99,102,241,0.55)',
          'shadow-opacity': 1,
        }}
      }},
      /* --- outside edges --- */
      {{
        selector: 'edge[edge_type="outside"]',
        style: {{
          'width': 1,
          'line-color': 'rgba(71,85,105,0.25)',
          'target-arrow-color': 'rgba(71,85,105,0.25)',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          'arrow-scale': 0.7,
        }}
      }},
      /* --- subgraph edges --- */
      {{
        selector: 'edge[edge_type="subgraph"]',
        style: {{
          'width': 1.8,
          'line-color': 'rgba(148,163,184,0.35)',
          'target-arrow-color': 'rgba(148,163,184,0.45)',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          'arrow-scale': 0.9,
        }}
      }},
      /* --- plan edges: bright amber path --- */
      {{
        selector: 'edge[edge_type="plan"]',
        style: {{
          'width': 3.5,
          'line-color': '#f59e0b',
          'target-arrow-color': '#f59e0b',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          'arrow-scale': 1.3,
          'shadow-blur': 8,
          'shadow-color': 'rgba(245,158,11,0.4)',
          'shadow-opacity': 1,
        }}
      }},
      /* --- animation classes --- */
      {{
        selector: '.anim-pulse',
        style: {{
          'border-color': '#fbbf24',
          'border-width': 4,
          'shadow-blur': 30,
          'shadow-color': 'rgba(251,191,36,0.7)',
          'shadow-opacity': 1,
          'z-index': 20,
        }}
      }},
      {{
        selector: '.anim-edge',
        style: {{
          'line-color': '#fbbf24',
          'target-arrow-color': '#fbbf24',
          'width': 5,
          'shadow-blur': 14,
          'shadow-color': 'rgba(251,191,36,0.6)',
          'shadow-opacity': 1,
          'z-index': 20,
        }}
      }},
    ],
    layout: {{
      name: 'breadthfirst',
      directed: true,
      spacingFactor: 1.3,
      avoidOverlap: true,
      padding: 35,
      roots: elements
        .filter(function(e) {{ return e.data && !e.data.source; }})
        .filter(function(e) {{
          var id = e.data.id;
          var isTarget = elements.some(function(edge) {{
            return edge.data && edge.data.target === id;
          }});
          return !isTarget;
        }})
        .map(function(e) {{ return e.data.id; }}),
    }},
    userZoomingEnabled: true,
    userPanningEnabled: true,
    boxSelectionEnabled: false,
    minZoom: 0.4,
    maxZoom: 2.5,
  }});

  /* ---- Tooltip ---- */
  var tooltip = document.getElementById('cy-tooltip');
  var container = document.getElementById('cy-graph');

  cy.on('mouseover', 'node', function(e) {{
    var d = e.target.data();
    var statusMap = {{
      target: '<span style="color:#818cf8">Целевая тема</span>',
      plan: '<span style="color:#fbbf24">В плане (шаг ' + d.plan_order + ')</span>',
      mastered: '<span style="color:#34d399">Освоено</span>',
      subgraph: 'В подграфе',
      outside: '<span style="opacity:0.6">Вне подграфа</span>'
    }};
    var mColor = masteryColor(d.mastery);
    var bar = '<div style="margin:6px 0 4px;height:6px;border-radius:3px;' +
      'background:#1e293b;overflow:hidden;">' +
      '<div style="width:' + (d.mastery * 100) + '%;height:100%;border-radius:3px;' +
      'background:' + mColor + ';"></div></div>';
    tooltip.innerHTML =
      '<div style="font-weight:600;font-size:14px;margin-bottom:2px;">' +
      d.label + '</div>' +
      '<div style="font-size:11px;color:#94a3b8;margin-bottom:3px;">' +
      (d.subject || '') + ' · ' + d.grade + ' класс</div>' +
      '<div style="font-size:12px;margin-bottom:4px;">' +
      (statusMap[d.node_type] || '') + '</div>' +
      '<div style="font-size:12px;">Mastery: <b style="color:' +
      mColor + '">' + d.mastery_pct + '</b></div>' + bar;
    tooltip.style.display = 'block';
  }});

  cy.on('mousemove', 'node', function(e) {{
    var rect = container.getBoundingClientRect();
    var x = e.originalEvent.clientX - rect.left + 15;
    var y = e.originalEvent.clientY - rect.top - 10;
    if (x + 230 > rect.width) x = x - 245;
    if (y + 120 > rect.height) y = y - 100;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }});

  cy.on('mouseout', 'node', function() {{
    tooltip.style.display = 'none';
  }});

  /* ---- Animate plan path step by step ---- */
  var planIds = {plan_ids};
  var animStep = 0;
  function nextStep() {{
    if (animStep >= planIds.length) return;
    var nid = planIds[animStep];
    cy.getElementById(nid).addClass('anim-pulse');
    if (animStep > 0) {{
      var prev = planIds[animStep - 1];
      cy.edges().forEach(function(e) {{
        if (e.data('source') === prev && e.data('target') === nid)
          e.addClass('anim-edge');
      }});
    }}
    animStep++;
    if (animStep < planIds.length) setTimeout(nextStep, 700);
  }}
  setTimeout(nextStep, 900);
}});
</script>
"""
                components.html(cy_html, height=580)

                path_str = " -> ".join(KC_NAMES[kc] for kc in plan)
                st.markdown(f"**Порядок прохождения:** {path_str}")

                plan_rows = []
                for i, kc in enumerate(plan):
                    prereqs = KC_GRAPH.get(kc, [])
                    prereq_str = ", ".join(
                        f"{KC_NAMES[p]} ({vis_m.get(p, 0):.2f})" for p in prereqs
                    ) if prereqs else "--"
                    plan_rows.append({
                        "N": i + 1,
                        "Тема": KC_NAMES[kc],
                        "Текущий mastery": round(vis_m.get(kc, 0), 2),
                        "Цель": mastery_threshold,
                        "Пререквизиты": prereq_str,
                    })
                st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)

            st.divider()

            # ============================================================
            # ФАЗА 4: Обучение (итеративно по шагам плана)
            # ============================================================
            if plan:
                st.subheader("Фаза 4: Прохождение плана обучения")
                st.markdown(
                    "Для каждой темы Micro-уровень подбирает задания в 3 шага:  \n"
                    "**1. IRT Pre-filter** — отсекает слишком лёгкие/сложные "
                    "(P(correct) вне [0.20, 0.90]).  \n"
                    "**2. ZPD Target** — вычисляет целевую сложность "
                    "(build: mastery+0.10, consolidate: −0.10, test: +0.30).  \n"
                    "**3. select_task** — 80% exploitation (ближайшая к ZPD), "
                    "20% stretch exploration (сложнее уровня, для калибровки mastery).  \n"
                    "В полной системе к этому добавляются: "
                    "cluster exploration (малотестированные задания кластера) "
                    "и Thompson Sampling (байесовский выбор с cluster prior).  \n"
                    "Каждые 5 заданий Macro-уровень получает MicroSummary, "
                    "проводит диагностику и принимает решение."
                )

                history: list[dict] = []
                all_vis_m = dict(vis_m)
                mastery_snapshots: list[dict] = []
                consecutive_correct: dict[str, int] = {kc: 0 for kc in sim_kcs}
                recent_scores: dict[str, list[float]] = {kc: [] for kc in sim_kcs}

                for step_idx, plan_kc in enumerate(plan):
                    st.markdown(f"---")
                    st.markdown(
                        f"#### Шаг {step_idx + 1}/{len(plan)}: {KC_NAMES[plan_kc]}  \n"
                        f"Начальный mastery: **{all_vis_m.get(plan_kc, 0):.2f}**, "
                        f"цель: **{mastery_threshold}**"
                    )

                    step_history: list[dict] = []
                    max_tasks = 15
                    tasks_done = 0
                    final_action = "continue"

                    while tasks_done < max_tasks:
                        kc_tasks = [t for t in TASK_POOL if t["kc_id"] == plan_kc]
                        filtered, is_fallback = filter_tasks_by_irt(
                            kc_tasks, all_vis_m.get(plan_kc, 0.5),
                        )
                        target_diff = compute_zpd_target_difficulty(
                            all_vis_m.get(plan_kc, 0.5), "build",
                        )
                        task, source = select_task(
                            filtered, target_diff,
                            mastery=all_vis_m.get(plan_kc, 0.5),
                            rng=rng,
                        )
                        irt_diff = task["parts"][0]["irt_difficulty"]
                        score = simulate_answer(
                            true_m[plan_kc], irt_diff, p_slip, p_guess, rng,
                        )

                        old_m = all_vis_m[plan_kc]
                        ra = (
                            sum(recent_scores[plan_kc][-5:])
                            / max(1, len(recent_scores[plan_kc][-5:]))
                        )
                        all_vis_m[plan_kc] = smooth_update(
                            all_vis_m[plan_kc], score,
                            consecutive_correct=consecutive_correct[plan_kc],
                            recent_accuracy=ra,
                            irt_difficulty=irt_diff,
                        )
                        delta = all_vis_m[plan_kc] - old_m
                        true_m[plan_kc] = min(
                            1.0,
                            true_m[plan_kc]
                            + growth * score * (1.0 - true_m[plan_kc]),
                        )
                        if score >= 0.9:
                            consecutive_correct[plan_kc] += 1
                        else:
                            consecutive_correct[plan_kc] = 0
                        recent_scores[plan_kc].append(score)

                        tasks_done += 1
                        source_ru = {"zpd": "ZPD", "stretch": "stretch", "exploration": "exploration"}.get(source, source)
                        row = {
                            "N": tasks_done,
                            "Задание": task["task_id"],
                            "Источник": source_ru,
                            "Сложность": diff_label(irt_diff),
                            "Ответ": score_label(score),
                            "Mastery": f"{all_vis_m[plan_kc]:.3f}",
                            "delta": f"{delta:+.3f}",
                        }
                        step_history.append(row)
                        history.append({
                            "kc_id": plan_kc, "score": score,
                            "irt_diff": irt_diff, "source": source,
                            "vis_mastery": all_vis_m[plan_kc],
                            "true_mastery": true_m[plan_kc],
                            "mastery_delta": delta,
                        })
                        mastery_snapshots.append({
                            "step": len(mastery_snapshots) + 1,
                            "kc": KC_NAMES[plan_kc],
                            "vis": all_vis_m[plan_kc],
                            "true": true_m[plan_kc],
                        })

                        # Триггер MicroSummary каждые 5 заданий или при достижении порога
                        if tasks_done % 5 == 0 or all_vis_m[plan_kc] >= mastery_threshold:
                            summary = compute_micro_summary(
                                history, plan_kc, all_vis_m[plan_kc],
                            )
                            prereqs = KC_GRAPH.get(plan_kc, [])
                            weakest = min(
                                (all_vis_m.get(p, 0.5) for p in prereqs),
                                default=None,
                            )
                            n_att = len([
                                h for h in history if h["kc_id"] == plan_kc
                            ])
                            conf = compute_confidence(
                                attempts_count=n_att,
                                recent_accuracy=summary["avg_score"],
                                probability_effective=all_vis_m[plan_kc],
                                last_practiced=datetime(2025, 1, 1),
                                now=datetime(2025, 1, 1, 0, 1),
                            )
                            d = diagnose(
                                mastery_current=all_vis_m[plan_kc],
                                velocity=summary["velocity"],
                                frustration_count=summary["frustration_count"],
                                avg_score=summary["avg_score"],
                                tasks_spent=tasks_done,
                                attempts_count=n_att,
                                mastery_confidence=conf,
                                weakest_prereq_mastery=weakest,
                                task_count_for_kc=len(kc_tasks),
                            )

                            if all_vis_m[plan_kc] >= mastery_threshold:
                                final_action = "advance"
                            elif d.reason == "prereq_gap":
                                final_action = "insert_prereq"
                            elif summary["frustration_count"] >= 3:
                                final_action = "consolidate"
                            else:
                                final_action = "continue"

                            # -- Отображение: таблица заданий до этого момента --
                            st.markdown(
                                f"**История ответов (задания 1-{tasks_done}):**"
                            )
                            st.dataframe(
                                pd.DataFrame(step_history),
                                use_container_width=True,
                                hide_index=True,
                            )

                            # -- Micro -> Macro коммуникация --
                            st.info(
                                f"**[MICRO -> MicroSummary]**  \n"
                                f"avg_score = {summary['avg_score']:.2f}, "
                                f"velocity = {summary['velocity']:+.3f}, "
                                f"frustration = {summary['frustration_count']}, "
                                f"confidence = {conf:.2f}"
                            )

                            label_d, desc_d = DIAGNOSIS_RU[d.reason]
                            diag_msg = (
                                f"**[MACRO <- Диагностика]** {label_d} "
                                f"(уверенность {d.confidence:.0%}). {desc_d}."
                            )

                            if final_action == "advance":
                                st.success(
                                    f"{diag_msg}  \n"
                                    f"Mastery **{all_vis_m[plan_kc]:.2f}** >= "
                                    f"{mastery_threshold} -- "
                                    f"**ADVANCE: переход к следующей теме**"
                                )
                                break
                            elif final_action == "insert_prereq":
                                st.warning(
                                    f"{diag_msg}  \n"
                                    f"**INSERT_PREREQ: рекомендуется вставить "
                                    f"дополнительный пререквизит**"
                                )
                                break
                            elif final_action == "consolidate":
                                st.warning(
                                    f"{diag_msg}  \n"
                                    f"**CONSOLIDATE: переход в режим закрепления "
                                    f"(снижение сложности)**"
                                )
                                break
                            else:
                                st.info(
                                    f"{diag_msg}  \n"
                                    f"**CONTINUE: продолжаем обучение**"
                                )

                    # Если бюджет исчерпан без решения
                    if final_action == "continue" and tasks_done >= max_tasks:
                        st.markdown(
                            f"**История ответов (задания 1-{tasks_done}):**"
                        )
                        st.dataframe(
                            pd.DataFrame(step_history),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.warning(
                            f"**[MACRO]** Бюджет исчерпан ({tasks_done} заданий). "
                            f"Mastery: {all_vis_m[plan_kc]:.2f}. "
                            f"Переход к следующей теме."
                        )

                st.divider()

                # ============================================================
                # ФАЗА 5: Итоги
                # ============================================================
                st.subheader("Итоги обучения")

                scores_all = [h["score"] for h in history]
                mastered_kcs = [kc for kc in plan if all_vis_m.get(kc, 0) >= mastery_threshold]
                failed_kcs = [kc for kc in plan if all_vis_m.get(kc, 0) < mastery_threshold]

                m1, m2, m3 = st.columns(3)
                m1.metric("Всего заданий", len(history))
                acc = sum(1 for s in scores_all if s >= 0.9) / len(scores_all)
                m2.metric("Точность", f"{acc:.0%}")
                m3.metric("Средний балл", f"{sum(scores_all) / len(scores_all):.2f}")

                m4, m5, m6 = st.columns(3)
                m4.metric("Всего тем", len(plan))
                m5.metric("Освоено тем", len(mastered_kcs),
                          delta=f"{len(mastered_kcs)}/{len(plan)}", delta_color="normal")
                m6.metric("Проблемные темы", len(failed_kcs),
                          delta=f"{len(failed_kcs)}" if failed_kcs else "нет",
                          delta_color="inverse")

                if failed_kcs:
                    failed_rows = []
                    for kc in failed_kcs:
                        kc_hist = [h for h in history if h["kc_id"] == kc]
                        avg_s = sum(h["score"] for h in kc_hist) / max(1, len(kc_hist))
                        failed_rows.append({
                            "Тема": KC_NAMES[kc],
                            "Финальный mastery": f"{all_vis_m.get(kc, 0):.2f}",
                            "Порог": mastery_threshold,
                            "Заданий": len(kc_hist),
                            "Средний балл": f"{avg_s:.2f}",
                            "Причина": "бюджет исчерпан" if len(kc_hist) >= 15 else "macro-решение",
                        })
                    st.markdown("**Проблемные темы (не достигли порога):**")
                    st.dataframe(pd.DataFrame(failed_rows), use_container_width=True, hide_index=True)

                st.markdown("**Эволюция mastery по шагам обучения:**")
                df_snap = pd.DataFrame(mastery_snapshots)
                fig_evo = go.Figure()
                for kc_name in df_snap["kc"].unique():
                    subset = df_snap[df_snap["kc"] == kc_name]
                    fig_evo.add_trace(go.Scatter(
                        x=subset["step"], y=subset["vis"],
                        name=f"{kc_name} (система)", mode="lines",
                    ))
                    fig_evo.add_trace(go.Scatter(
                        x=subset["step"], y=subset["true"],
                        name=f"{kc_name} (истинный)", mode="lines",
                        line=dict(dash="dot"),
                    ))
                fig_evo.add_hline(
                    y=mastery_threshold, line_dash="dash",
                    line_color="gray",
                    annotation_text=f"порог {mastery_threshold}",
                )
                fig_evo.update_layout(
                    xaxis_title="Шаг", yaxis_title="Mastery",
                    yaxis_range=[0, 1.05], height=420,
                    legend=dict(font=dict(size=10)),
                )
                st.plotly_chart(fig_evo, use_container_width=True)


# ===================================================================
# TAB 2 -- Micro-уровень
# ===================================================================
with tab_micro:
    st.header("Micro-уровень: как система подбирает задание")
    st.markdown(
        "Полный пайплайн подбора задания:  \n"
        "1. **IRT Pre-filter** — отсекает задания, где P(correct) вне [0.20, 0.90]  \n"
        "2. **ZPD target** — вычисляет целевую сложность по режиму "
        "(build +0.10, consolidate −0.10, test +0.30)  \n"
        "3. **select_task** — 80% exploitation (ближайшее к ZPD target), "
        "20% stretch/exploration  \n"
        "4. **Thompson Sampling** — байесовский бандит уточняет выбор на основе "
        "контекстного вектора (13 фич)  \n"
        "5. **Cluster exploration** — 20% шанс (в build) попробовать задание из "
        "кластерной статистики  \n\n"
        "Режим определяется автоматически: **build** → **test** (mastery близко к порогу) "
        "→ **consolidate** (при низкой точности ответов)."
    )
    with st.expander("Подробнее о режимах обучения"):
        st.markdown(
            "Система автоматически переключает режим на основе прогресса ученика.  \n"
            "Каждый режим меняет **три параметра**: целевую сложность, "
            "границы IRT-фильтра и уровень exploration.\n\n"
            "| Режим | Когда включается | ZPD сдвиг | IRT диапазон | Cluster explore | ε (случайность) |\n"
            "|-------|-----------------|-----------|--------------|----------------|------------------|\n"
            "| **Build** | По умолчанию — ученик изучает тему | +0.10 (чуть сложнее) | [0.20, 0.90] | 20% | 5% |\n"
            "| **Consolidate** | Точность < 45% за последние 5 заданий или фрустрация ≥ 2 | −0.10 (чуть легче) | [0.40, 0.95] | 10% | 10% |\n"
            "| **Test** | Mastery ≥ порог − 0.10 (близко к цели) | +0.30 (заметно сложнее) | [0.15, 0.85] | 0% | 0% |\n\n"
            "**Build** — основной режим: система даёт задания чуть выше текущего уровня, "
            "активно исследует через кластерную статистику (20%).  \n"
            "**Consolidate** — включается при трудностях: система снижает сложность, "
            "расширяет допустимый диапазон P(correct) вверх до 0.95, "
            "увеличивает случайный выбор (ε=10%) чтобы найти задания, которые подходят лучше.  \n"
            "**Test** — финальная проверка: задания значительно сложнее, "
            "никакого exploration — чистая эксплуатация лучшего задания для проверки знаний."
        )

    col_mi_cfg, col_mi_viz = st.columns([1, 3])

    with col_mi_cfg:
        st.subheader("Профиль")
        mi_grade = st.slider("Класс", 5, 11, 8, key="mi_grade")
        mi_type = st.selectbox("Тип ученика", list(STUDENT_TYPES.keys()), key="mi_type")

        mi_kcs = grade_kcs(mi_grade)
        mi_kc = st.selectbox(
            "Тема для изучения", mi_kcs,
            format_func=lambda x: f"{KC_NAMES[x]} ({KC_INTRO_GRADE[x]} кл.)",
            key="mi_kc",
        )
        mi_mastery_init = st.slider("Начальный mastery", 0.10, 0.80, 0.35, 0.05, key="mi_m0")
        mastery_threshold_mi = st.slider("Порог освоения", 0.60, 0.95, 0.80, 0.05, key="mi_thr")
        mi_n_tasks = st.slider("Число заданий", 5, 30, 15, key="mi_n")

        mi_view = st.radio("Режим просмотра", ["Полный путь", "Пошагово"], key="mi_view")

    def _auto_mode(mastery: float, threshold: float, recent: list[float]) -> str:
        if mastery >= threshold - 0.10:
            return "test"
        ra = sum(recent[-5:]) / max(1, len(recent[-5:])) if recent else 1.0
        if len(recent) >= 3 and ra < 0.45:
            return "consolidate"
        return "build"

    with col_mi_viz:
        mi_params = STUDENT_TYPES[mi_type]
        mi_growth = mi_params["growth"]
        mi_true_m_init = init_true_mastery(mi_grade, mi_params["mod"])[mi_kc]

        if mi_view == "Полный путь":
            run_mi = st.button("Показать полный путь", type="primary", key="mi_run")
            if not run_mi:
                st.info("Нажмите **Показать полный путь** — система подберёт все задания сразу.")
            else:
                mi_rng = random.Random(42)
                cur_m = mi_mastery_init
                cur_true = mi_true_m_init
                mi_consec = 0
                mi_recent: list[float] = []
                mi_log: list[dict] = []

                for step in range(mi_n_tasks):
                    mode = _auto_mode(cur_m, mastery_threshold_mi, mi_recent)
                    kc_tasks = [t for t in TASK_POOL if t["kc_id"] == mi_kc]
                    filtered, fb = filter_tasks_by_irt(kc_tasks, cur_m)
                    target_d = compute_zpd_target_difficulty(cur_m, mode)
                    task, source = select_task(filtered, target_d, mastery=cur_m, rng=mi_rng)
                    irt_diff = task["parts"][0]["irt_difficulty"]
                    p_pred = compute_p_correct(cur_m, irt_diff)
                    score = simulate_answer(cur_true, irt_diff, mi_params["p_slip"], mi_params["p_guess"], mi_rng)

                    old_m = cur_m
                    ra = sum(mi_recent[-5:]) / max(1, len(mi_recent[-5:]))
                    cur_m = smooth_update(cur_m, score, consecutive_correct=mi_consec, recent_accuracy=ra, irt_difficulty=irt_diff)
                    delta = cur_m - old_m
                    mi_consec = mi_consec + 1 if score >= 0.9 else 0
                    mi_recent.append(score)
                    cur_true = min(1.0, cur_true + mi_growth * score * (1.0 - cur_true))

                    mode_ru = {"build": "build", "consolidate": "consolid.", "test": "test"}[mode]
                    mi_log.append({
                        "N": step + 1,
                        "Задание": task["task_id"],
                        "Режим": mode_ru,
                        "IRT diff": round(irt_diff, 2),
                        "P(correct)": f"{p_pred:.2f}",
                        "ZPD target": f"{target_d:.2f}",
                        "Источник": source,
                        "Ответ": score_label(score),
                        "Mastery": f"{cur_m:.3f}",
                        "Истинный": f"{cur_true:.3f}",
                        "delta": f"{delta:+.3f}",
                    })

                st.dataframe(pd.DataFrame(mi_log), use_container_width=True, hide_index=True)

                mastery_vals = [mi_mastery_init] + [float(r["Mastery"]) for r in mi_log]
                true_vals = [mi_true_m_init] + [float(r["Истинный"]) for r in mi_log]
                fig_mi = go.Figure()
                fig_mi.add_trace(go.Scatter(
                    x=list(range(len(mastery_vals))), y=mastery_vals,
                    name="Mastery (оценка системы)", mode="lines+markers",
                    line=dict(color="#636EFA", width=2),
                ))
                fig_mi.add_trace(go.Scatter(
                    x=list(range(len(true_vals))), y=true_vals,
                    name="Истинный mastery", mode="lines+markers",
                    line=dict(color="#00CC96", width=2, dash="dash"),
                ))
                fig_mi.add_hline(y=mastery_threshold_mi, line_dash="dot", line_color="#EF553B",
                                 annotation_text=f"порог ({mastery_threshold_mi:.2f})")
                fig_mi.update_layout(
                    xaxis_title="Шаг", yaxis_title="Mastery",
                    yaxis_range=[0, 1.05], height=320,
                )
                st.plotly_chart(fig_mi, use_container_width=True)

                irt_diffs = [float(r["IRT diff"]) for r in mi_log]
                sources = [r["Источник"] for r in mi_log]
                modes = [r["Режим"] for r in mi_log]
                colors = ["#636EFA" if s == "zpd" else "#EF553B" if s == "stretch" else "#FFA15A" for s in sources]
                fig_diff = go.Figure()
                fig_diff.add_trace(go.Bar(
                    x=list(range(1, len(irt_diffs) + 1)), y=irt_diffs,
                    marker_color=colors,
                    text=[f"{s}<br>{m}" for s, m in zip(sources, modes)],
                    textposition="outside",
                ))
                fig_diff.update_layout(
                    xaxis_title="Шаг", yaxis_title="IRT difficulty",
                    height=300,
                    title="Сложность заданий (цвет = источник, текст = режим)",
                )
                st.plotly_chart(fig_diff, use_container_width=True)

        else:
            ss = st.session_state
            if "mi_step_state" not in ss or st.button("Сбросить", key="mi_reset"):
                ss.mi_step_state = {
                    "mastery": mi_mastery_init,
                    "true_m": mi_true_m_init,
                    "consec": 0,
                    "recent": [],
                    "log": [],
                    "rng_state": 42,
                }

            state = ss.mi_step_state
            step_n = len(state["log"])

            if step_n > 0:
                st.dataframe(pd.DataFrame(state["log"]), use_container_width=True, hide_index=True)

                mastery_vals = [mi_mastery_init] + [float(r["Mastery"]) for r in state["log"]]
                true_vals = [mi_true_m_init] + [float(r["Истинный"]) for r in state["log"]]
                fig_step = go.Figure()
                fig_step.add_trace(go.Scatter(
                    x=list(range(len(mastery_vals))), y=mastery_vals,
                    name="Mastery (оценка)", mode="lines+markers",
                    line=dict(color="#636EFA", width=2),
                ))
                fig_step.add_trace(go.Scatter(
                    x=list(range(len(true_vals))), y=true_vals,
                    name="Истинный", mode="lines+markers",
                    line=dict(color="#00CC96", width=2, dash="dash"),
                ))
                fig_step.add_hline(y=mastery_threshold_mi, line_dash="dot", line_color="#EF553B",
                                   annotation_text=f"порог ({mastery_threshold_mi:.2f})")
                fig_step.update_layout(
                    xaxis_title="Шаг", yaxis_title="Mastery",
                    yaxis_range=[0, 1.05], height=280,
                )
                st.plotly_chart(fig_step, use_container_width=True)

            if step_n >= mi_n_tasks:
                st.success(f"Все {mi_n_tasks} заданий пройдены. Финальный mastery: **{state['mastery']:.3f}** (истинный: **{state['true_m']:.3f}**)")
            else:
                mode = _auto_mode(state["mastery"], mastery_threshold_mi, state["recent"])
                mi_rng = random.Random(state["rng_state"])
                kc_tasks = [t for t in TASK_POOL if t["kc_id"] == mi_kc]
                filtered, fb = filter_tasks_by_irt(kc_tasks, state["mastery"])
                target_d = compute_zpd_target_difficulty(state["mastery"], mode)
                task, source = select_task(filtered, target_d, mastery=state["mastery"], rng=mi_rng)
                irt_diff = task["parts"][0]["irt_difficulty"]
                p_pred = compute_p_correct(state["mastery"], irt_diff)

                mode_labels = {"build": "Build (обучение)", "consolidate": "Consolidate (закрепление)", "test": "Test (проверка)"}
                st.markdown(f"**Шаг {step_n + 1}** — режим: **{mode_labels[mode]}**")
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Задание", task["task_id"].split("_t")[1])
                pc2.metric("IRT difficulty", f"{irt_diff:.2f}")
                pc3.metric("P(correct)", f"{p_pred:.2f}")
                pc4.metric("Метод", source)

                st.caption(
                    f"IRT-фильтр: {len(filtered)}/{len(kc_tasks)} заданий прошли"
                    f"{' (fallback)' if fb else ''}. "
                    f"ZPD target: {target_d:.2f}. Mastery: {state['mastery']:.3f}. "
                    f"Истинный: {state['true_m']:.3f}"
                )

                if st.button("Следующее задание (ответить и перейти)", type="primary", key="mi_next"):
                    score = simulate_answer(state["true_m"], irt_diff, mi_params["p_slip"], mi_params["p_guess"], mi_rng)
                    old_m = state["mastery"]
                    ra = sum(state["recent"][-5:]) / max(1, len(state["recent"][-5:]))
                    state["mastery"] = smooth_update(old_m, score, consecutive_correct=state["consec"], recent_accuracy=ra, irt_difficulty=irt_diff)
                    delta = state["mastery"] - old_m
                    state["consec"] = state["consec"] + 1 if score >= 0.9 else 0
                    state["recent"].append(score)
                    state["true_m"] = min(1.0, state["true_m"] + mi_growth * score * (1.0 - state["true_m"]))
                    state["rng_state"] = mi_rng.randint(0, 2**31)
                    mode_ru = {"build": "build", "consolidate": "consolid.", "test": "test"}[mode]
                    state["log"].append({
                        "N": step_n + 1,
                        "Задание": task["task_id"],
                        "Режим": mode_ru,
                        "IRT diff": round(irt_diff, 2),
                        "P(correct)": f"{p_pred:.2f}",
                        "ZPD target": f"{target_d:.2f}",
                        "Источник": source,
                        "Ответ": score_label(score),
                        "Mastery": f"{state['mastery']:.3f}",
                        "Истинный": f"{state['true_m']:.3f}",
                        "delta": f"{delta:+.3f}",
                    })
                    st.rerun()


# ===================================================================
# TAB 3 -- Macro-уровень
# ===================================================================
with tab_macro:
    st.header("Macro-уровень: построение плана обучения")
    st.markdown(
        "Macro-уровень строит **учебный план** на основе профиля ученика: "
        "определяет порядок тем, оценивает бюджет заданий и риски для каждого шага. "
        "Профиль влияет на строгость пререквизитов, бюджет заданий и готовность к тесту."
    )

    col_ma_cfg, col_ma_viz = st.columns([1, 3])

    with col_ma_cfg:
        st.subheader("Профиль ученика")

        ma_presets = {
            "Средний ученик": dict(
                uncertainty=0.40, confidence=0.50, weak_prereq=0.25,
                speed_global=0.06, speed_recent=0.06, recovery=0.55,
                frustration=0.30, stall_base=0.25, regression_base=0.20,
                pacing="normal", budget_mult=1.0, prereq_strict=0.70,
                test_bias=0.0, granularity=1.0,
            ),
            "Быстрый ученик": dict(
                uncertainty=0.20, confidence=0.70, weak_prereq=0.10,
                speed_global=0.12, speed_recent=0.14, recovery=0.75,
                frustration=0.15, stall_base=0.10, regression_base=0.10,
                pacing="fast", budget_mult=0.7, prereq_strict=0.60,
                test_bias=0.15, granularity=0.8,
            ),
            "Медленный ученик": dict(
                uncertainty=0.60, confidence=0.30, weak_prereq=0.45,
                speed_global=0.03, speed_recent=0.02, recovery=0.35,
                frustration=0.55, stall_base=0.50, regression_base=0.40,
                pacing="slow", budget_mult=1.5, prereq_strict=0.80,
                test_bias=-0.10, granularity=1.3,
            ),
            "Настроить вручную": None,
        }
        ma_preset = st.selectbox("Пресет", list(ma_presets.keys()), key="ma_preset")
        p = ma_presets[ma_preset] or ma_presets["Средний ученик"]

        with st.expander("Параметры профиля", expanded=ma_preset == "Настроить вручную"):
            ma_uncertainty = st.slider("Неопределённость оценки", 0.0, 1.0, p["uncertainty"], 0.05, key="ma_unc")
            ma_confidence = st.slider("Средняя confidence", 0.0, 1.0, p["confidence"], 0.05, key="ma_conf")
            ma_weak_prereq = st.slider("Доля слабых пререквизитов", 0.0, 1.0, p["weak_prereq"], 0.05, key="ma_wp")
            ma_speed = st.slider("Скорость обучения (global)", 0.01, 0.20, p["speed_global"], 0.01, key="ma_spd")
            ma_recovery = st.slider("Восстановление после ошибки", 0.0, 1.0, p["recovery"], 0.05, key="ma_rec")
            ma_frustration = st.slider("Риск фрустрации", 0.0, 1.0, p["frustration"], 0.05, key="ma_frust")
            ma_stall = st.slider("Базовый риск застревания", 0.0, 1.0, p["stall_base"], 0.05, key="ma_stall")
            ma_regression = st.slider("Базовый риск регрессии", 0.0, 1.0, p["regression_base"], 0.05, key="ma_regr")
            ma_budget = st.slider("Множитель бюджета", 0.5, 2.0, p["budget_mult"], 0.1, key="ma_budg")
            ma_prereq_strict = st.slider("Строгость пререквизитов", 0.50, 0.95, p["prereq_strict"], 0.05, key="ma_ps")

        st.divider()
        ma_grade = st.slider("Класс", 5, 11, 8, key="ma_grade")
        ma_kcs = grade_kcs(ma_grade)
        ma_target = st.selectbox(
            "Целевая тема", ma_kcs,
            format_func=lambda x: f"{KC_NAMES[x]} ({KC_INTRO_GRADE[x]} кл.)",
            key="ma_target",
        )
        ma_threshold = st.slider("Порог освоения", 0.60, 0.95, 0.80, 0.05, key="ma_thr")

        ma_type = st.selectbox("Тип ученика (mastery)", list(STUDENT_TYPES.keys()), key="ma_st_type")

    with col_ma_viz:
        import json as _json
        import streamlit.components.v1 as components

        ma_st_params = STUDENT_TYPES[ma_type]
        ma_true_m = init_true_mastery(ma_grade, ma_st_params["mod"])

        profile = MacroStudentProfile(
            student_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
            updated_at=datetime.now(),
            target_kc_id=ma_target,
            uncertainty_level=ma_uncertainty,
            mastery_confidence_mean=ma_confidence,
            weak_prereq_fraction=ma_weak_prereq,
            target_subgraph_mastery_mean=0.40,
            learning_speed_global=ma_speed,
            learning_speed_recent=ma_speed,
            tasks_to_gain_01_mastery=round(1.0 / max(0.01, ma_speed)),
            recovery_after_error=ma_recovery,
            frustration_risk=ma_frustration,
            stall_risk_baseline=ma_stall,
            regression_risk_baseline=ma_regression,
            pacing_mode=p["pacing"],
            budget_multiplier=ma_budget,
            prereq_strictness=ma_prereq_strict,
            test_readiness_bias=p["test_bias"],
            step_granularity=p["granularity"],
        )

        plan = build_learning_path(ma_target, ma_true_m, ma_threshold)

        if not plan:
            st.success(
                f"Ученик уже освоил **{KC_NAMES[ma_target]}** и все пререквизиты "
                f"(mastery >= {ma_threshold})!"
            )
        else:
            def _collect_subgraph_ma(target: str) -> set[str]:
                nodes: set[str] = set()
                def _walk(kc: str) -> None:
                    if kc in nodes:
                        return
                    nodes.add(kc)
                    for pr in KC_GRAPH.get(kc, []):
                        _walk(pr)
                _walk(target)
                return nodes

            subgraph_kcs = _collect_subgraph_ma(ma_target)
            plan_set = set(plan)

            plan_details = []
            for idx, kc in enumerate(plan):
                m = ma_true_m.get(kc, 0.0)
                prereqs = KC_GRAPH.get(kc, [])
                weak_pf = sum(1 for pr in prereqs if ma_true_m.get(pr, 0) < ma_prereq_strict) / max(1, len(prereqs))
                tasks_budget = estimate_tasks_to_mastery(
                    profile=profile, kc_id=kc,
                    current_mastery=m, target_mastery=ma_threshold,
                    weak_prereq_fraction=weak_pf,
                    confidence=ma_confidence,
                )
                stall = estimate_stall_risk(
                    profile=profile, kc_id=kc,
                    weak_prereq_fraction=weak_pf,
                    confidence=ma_confidence,
                    graph_depth=idx,
                )
                regr = estimate_regression_risk(
                    profile=profile, kc_id=kc,
                    confidence=ma_confidence,
                )
                is_target = kc == ma_target
                mode = "test" if m >= ma_threshold - 0.10 else "build"

                reasons = []
                if is_target:
                    reasons.append("целевая тема")
                else:
                    reasons.append("пререквизит")
                if m < 0.20:
                    reasons.append("mastery очень низкий")
                elif m < ma_threshold:
                    reasons.append(f"mastery {m:.0%} < порога {ma_threshold:.0%}")
                if weak_pf > 0.3:
                    reasons.append(f"слабые пререквизиты ({weak_pf:.0%})")
                if stall > 0.4:
                    reasons.append(f"высокий риск застревания")

                plan_details.append({
                    "Шаг": idx + 1,
                    "Тема": KC_NAMES[kc],
                    "Mastery": f"{m:.0%}",
                    "Режим": mode,
                    "Бюджет": tasks_budget,
                    "Stall risk": f"{stall:.0%}",
                    "Regr. risk": f"{regr:.0%}",
                    "Почему в плане": "; ".join(reasons),
                })

            st.subheader(f"План: {len(plan)} шагов к «{KC_NAMES[ma_target]}»")
            st.dataframe(pd.DataFrame(plan_details), use_container_width=True, hide_index=True)

            total_budget = sum(d["Бюджет"] for d in plan_details)
            avg_stall = np.mean([float(d["Stall risk"].strip("%")) / 100 for d in plan_details])
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Шагов в плане", len(plan))
            mc2.metric("Всего заданий", total_budget)
            mc3.metric("Ср. stall risk", f"{avg_stall:.0%}")
            mc4.metric("Pacing", p["pacing"])

            cy_nodes = []
            cy_edges = []
            for kc in ma_kcs:
                m_val = ma_true_m.get(kc, 0)
                in_subgraph = kc in subgraph_kcs
                if kc == ma_target:
                    node_type = "target"
                elif kc in plan_set:
                    node_type = "plan"
                elif in_subgraph and m_val >= ma_threshold:
                    node_type = "mastered"
                elif in_subgraph:
                    node_type = "subgraph"
                else:
                    node_type = "outside"
                plan_order = plan.index(kc) + 1 if kc in plan_set else 0
                subj = KC_SUBJECTS.get(kc, "")
                subj_ru = SUBJECT_RU.get(subj, subj)
                cy_nodes.append({
                    "data": {
                        "id": kc,
                        "label": KC_NAMES[kc],
                        "mastery": round(m_val, 2),
                        "mastery_pct": f"{m_val:.0%}",
                        "node_type": node_type,
                        "plan_order": plan_order,
                        "in_subgraph": in_subgraph,
                        "subject": subj_ru,
                        "grade": KC_INTRO_GRADE.get(kc, 0),
                    }
                })

            ma_kcs_set = set(ma_kcs)
            for kc in ma_kcs:
                for prereq in KC_GRAPH.get(kc, []):
                    if prereq not in ma_kcs_set:
                        continue
                    is_plan_edge = prereq in plan_set and kc in plan_set
                    is_subgraph_edge = prereq in subgraph_kcs and kc in subgraph_kcs
                    if is_plan_edge:
                        etype = "plan"
                    elif is_subgraph_edge:
                        etype = "subgraph"
                    else:
                        etype = "outside"
                    strength = EDGE_STRENGTHS.get((prereq, kc), 0.5)
                    cy_edges.append({
                        "data": {
                            "source": prereq,
                            "target": kc,
                            "edge_type": etype,
                            "strength": round(strength, 2),
                        }
                    })

            cy_elements = _json.dumps(cy_nodes + cy_edges)
            plan_ids = _json.dumps(plan)

            cy_html = f"""
<div style="position:relative;">
<div id="cy-graph-ma" style="width:100%;height:520px;border-radius:14px;
     background:#0e1117;position:relative;overflow:hidden;
     border:1px solid rgba(255,255,255,0.06);">
</div>
<div id="cy-tooltip-ma" style="position:absolute;display:none;padding:10px 14px;
     background:rgba(14,17,23,0.96);color:#e2e8f0;border-radius:10px;
     font-size:13px;pointer-events:none;z-index:100;
     border:1px solid rgba(99,102,241,0.35);
     box-shadow:0 8px 30px rgba(0,0,0,0.5);
     backdrop-filter:blur(12px);max-width:220px;"></div>
</div>

<div style="display:flex;gap:18px;margin-top:8px;padding:6px 2px;
     font-size:12.5px;color:#94a3b8;flex-wrap:wrap;align-items:center;">
  <span style="display:flex;align-items:center;gap:5px;">
    <span style="width:16px;height:16px;border-radius:4px;
      background:linear-gradient(135deg,#818cf8,#6366f1);
      box-shadow:0 0 10px rgba(99,102,241,0.5);"></span>
    Целевая тема</span>
  <span style="display:flex;align-items:center;gap:5px;">
    <span style="width:16px;height:16px;border-radius:4px;
      border:2px solid #f59e0b;background:rgba(245,158,11,0.15);"></span>
    Маршрут обучения</span>
  <span style="display:flex;align-items:center;gap:5px;">
    <span style="width:16px;height:16px;border-radius:4px;
      background:#10b981;"></span>
    Освоено</span>
  <span style="display:flex;align-items:center;gap:5px;">
    <span style="width:16px;height:16px;border-radius:4px;
      background:#334155;border:1px solid #475569;"></span>
    Пререквизит (в подграфе)</span>
</div>

<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {{

  function masteryColor(m) {{
    if (m >= 0.75) return '#10b981';
    if (m >= 0.50) return '#3b82f6';
    if (m >= 0.30) return '#f59e0b';
    return '#ef4444';
  }}

  var elements = {cy_elements};
  elements.forEach(function(e) {{
    if (e.data && !e.data.source && e.data.mastery !== undefined) {{
      var m = e.data.mastery;
      if (e.data.node_type === 'plan') {{
        e.data._bg = masteryColor(m);
      }}
    }}
  }});

  var cy = cytoscape({{
    container: document.getElementById('cy-graph-ma'),
    elements: elements,
    style: [
      {{
        selector: 'node[node_type="outside"]',
        style: {{
          'label': 'data(label)',
          'text-valign': 'center', 'text-halign': 'center',
          'font-size': '9px', 'color': '#64748b',
          'text-wrap': 'wrap', 'text-max-width': '75px',
          'background-color': '#1e293b',
          'width': 55, 'height': 55, 'shape': 'round-rectangle',
          'border-width': 1, 'border-color': '#334155', 'opacity': 0.5,
        }}
      }},
      {{
        selector: 'node[node_type="mastered"]',
        style: {{
          'label': 'data(label)',
          'text-valign': 'center', 'text-halign': 'center',
          'font-size': '10.5px', 'font-weight': '500', 'color': '#f0fdf4',
          'text-wrap': 'wrap', 'text-max-width': '85px',
          'background-color': '#10b981',
          'width': 66, 'height': 66, 'shape': 'round-rectangle',
          'border-width': 2, 'border-color': '#34d399',
        }}
      }},
      {{
        selector: 'node[node_type="subgraph"]',
        style: {{
          'label': 'data(label)',
          'text-valign': 'center', 'text-halign': 'center',
          'font-size': '10.5px', 'font-weight': '500', 'color': '#e2e8f0',
          'text-wrap': 'wrap', 'text-max-width': '85px',
          'background-color': '#334155',
          'width': 66, 'height': 66, 'shape': 'round-rectangle',
          'border-width': 2, 'border-color': '#64748b',
        }}
      }},
      {{
        selector: 'node[node_type="plan"]',
        style: {{
          'label': 'data(label)',
          'text-valign': 'center', 'text-halign': 'center',
          'font-size': '11px', 'font-weight': '600', 'color': '#fffbeb',
          'text-wrap': 'wrap', 'text-max-width': '90px',
          'background-color': 'data(_bg)',
          'width': 74, 'height': 74, 'shape': 'round-rectangle',
          'border-width': 3, 'border-color': '#f59e0b',
          'shadow-blur': 12, 'shadow-color': 'rgba(245,158,11,0.35)', 'shadow-opacity': 1,
        }}
      }},
      {{
        selector: 'node[node_type="target"]',
        style: {{
          'label': 'data(label)',
          'text-valign': 'center', 'text-halign': 'center',
          'font-size': '12px', 'font-weight': '700', 'color': '#eef2ff',
          'text-wrap': 'wrap', 'text-max-width': '95px',
          'background-color': '#6366f1',
          'width': 82, 'height': 82, 'shape': 'round-rectangle',
          'border-width': 3, 'border-color': '#818cf8',
          'shadow-blur': 25, 'shadow-color': 'rgba(99,102,241,0.5)', 'shadow-opacity': 1,
        }}
      }},
      {{
        selector: 'edge[edge_type="outside"]',
        style: {{
          'width': 1.5, 'line-color': '#334155',
          'target-arrow-color': '#334155', 'target-arrow-shape': 'triangle',
          'curve-style': 'bezier', 'arrow-scale': 0.7, 'opacity': 0.4,
        }}
      }},
      {{
        selector: 'edge[edge_type="subgraph"]',
        style: {{
          'width': 2, 'line-color': '#475569',
          'target-arrow-color': '#475569', 'target-arrow-shape': 'triangle',
          'curve-style': 'bezier', 'arrow-scale': 0.9,
        }}
      }},
      {{
        selector: 'edge[edge_type="plan"]',
        style: {{
          'width': 3.5, 'line-color': '#f59e0b',
          'target-arrow-color': '#f59e0b', 'target-arrow-shape': 'triangle',
          'curve-style': 'bezier', 'arrow-scale': 1.3,
          'shadow-blur': 8, 'shadow-color': 'rgba(245,158,11,0.4)', 'shadow-opacity': 1,
        }}
      }},
    ],
    layout: {{
      name: 'breadthfirst', directed: true,
      spacingFactor: 1.3, avoidOverlap: true, padding: 35,
      roots: elements
        .filter(function(e) {{ return e.data && !e.data.source; }})
        .filter(function(e) {{
          var id = e.data.id;
          return !elements.some(function(edge) {{ return edge.data && edge.data.target === id; }});
        }})
        .map(function(e) {{ return e.data.id; }}),
    }},
    userZoomingEnabled: true, userPanningEnabled: true,
    boxSelectionEnabled: false, minZoom: 0.4, maxZoom: 2.5,
  }});

  var tooltip = document.getElementById('cy-tooltip-ma');
  var container = document.getElementById('cy-graph-ma');

  cy.on('mouseover', 'node', function(e) {{
    var d = e.target.data();
    var statusMap = {{
      target: '<span style="color:#818cf8">Целевая тема</span>',
      plan: '<span style="color:#fbbf24">В плане (шаг ' + d.plan_order + ')</span>',
      mastered: '<span style="color:#34d399">Освоено</span>',
      subgraph: 'В подграфе',
      outside: '<span style="opacity:0.6">Вне подграфа</span>'
    }};
    var mColor = masteryColor(d.mastery);
    var bar = '<div style="margin:6px 0 4px;height:6px;border-radius:3px;' +
      'background:#1e293b;overflow:hidden;">' +
      '<div style="width:' + (d.mastery * 100) + '%;height:100%;border-radius:3px;' +
      'background:' + mColor + ';"></div></div>';
    tooltip.innerHTML =
      '<div style="font-weight:600;font-size:14px;margin-bottom:2px;">' + d.label + '</div>' +
      '<div style="font-size:11px;color:#94a3b8;margin-bottom:3px;">' +
      (d.subject || '') + ' · ' + d.grade + ' класс</div>' +
      '<div style="font-size:12px;margin-bottom:4px;">' + (statusMap[d.node_type] || '') + '</div>' +
      '<div style="font-size:12px;">Mastery: <b style="color:' + mColor + '">' + d.mastery_pct + '</b></div>' + bar;
    tooltip.style.display = 'block';
  }});

  cy.on('mousemove', 'node', function(e) {{
    var rect = container.getBoundingClientRect();
    var x = e.originalEvent.clientX - rect.left + 15;
    var y = e.originalEvent.clientY - rect.top - 10;
    if (x + 230 > rect.width) x = x - 245;
    if (y + 120 > rect.height) y = y - 100;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }});

  cy.on('mouseout', 'node', function() {{
    tooltip.style.display = 'none';
  }});

}});
</script>
"""
            components.html(cy_html, height=600)

            with st.expander("Как профиль влияет на план"):
                st.markdown(
                    f"**budget_multiplier = {ma_budget:.1f}** — "
                    f"{'увеличенный бюджет (больше заданий на каждый шаг)' if ma_budget > 1.1 else 'стандартный бюджет' if ma_budget > 0.9 else 'сокращённый бюджет (меньше заданий)'}  \n"
                    f"**prereq_strictness = {ma_prereq_strict:.2f}** — "
                    f"пререквизит считается слабым при mastery < {ma_prereq_strict:.0%}  \n"
                    f"**uncertainty = {ma_uncertainty:.2f}** — "
                    f"{'высокая неопределённость → больше заданий для надёжной оценки' if ma_uncertainty > 0.5 else 'умеренная неопределённость'}  \n"
                    f"**stall_risk_baseline = {ma_stall:.2f}** — "
                    f"{'высокий базовый риск застревания → система будет внимательнее к прогрессу' if ma_stall > 0.35 else 'низкий риск застревания'}  \n"
                    f"**pacing = {p['pacing']}** — "
                    f"{'медленный темп: больше повторений, выше строгость' if p['pacing'] == 'slow' else 'быстрый темп: меньше заданий, ранний переход к тесту' if p['pacing'] == 'fast' else 'стандартный темп'}"
                )


# ===================================================================
# TAB 4 -- Mastery и Confidence
# ===================================================================
with tab_mastery:
    st.header("Как система оценивает знания ученика")
    st.markdown(
        "**Mastery** -- оценка уровня знаний по теме (0 = не знает, 1 = знает). "
        "Обновляется после каждого ответа через EMA (экспоненциальное скользящее среднее).  \n"
        "**Confidence** -- насколько система доверяет своей оценке mastery. "
        "Зависит от количества попыток, стабильности результатов и давности практики."
    )

    col_sliders, col_charts = st.columns([1, 2])

    with col_sliders:
        st.subheader("Параметры")
        init_mastery = st.slider("Начальный mastery", 0.0, 1.0, 0.10, 0.01, key="bkt_init")
        lr = st.slider("Скорость обновления (lr)", 0.01, 0.50,
                        float(_bcfg.SMOOTH_LR), 0.01, key="bkt_lr")
        transit = st.slider("Transit (бонус за попытку)", 0.0, 0.10,
                            float(_bcfg.SMOOTH_TRANSIT), 0.005, key="bkt_transit")
        score_pattern = st.selectbox(
            "Паттерн ответов",
            [
                "Все правильные (1.0)",
                "Все неправильные (0.0)",
                "Чередование (1,0,1,0...)",
                "Постепенное улучшение",
                "Случайные",
            ],
            key="bkt_pattern",
        )
        n_steps = st.slider("Количество попыток", 5, 40, 20, key="bkt_steps")

    with col_charts:
        rng_bkt = random.Random(7)
        scores_seq: list[float] = []
        for i in range(n_steps):
            if score_pattern == "Все правильные (1.0)":
                scores_seq.append(1.0)
            elif score_pattern == "Все неправильные (0.0)":
                scores_seq.append(0.0)
            elif score_pattern == "Чередование (1,0,1,0...)":
                scores_seq.append(1.0 if i % 2 == 0 else 0.0)
            elif score_pattern == "Постепенное улучшение":
                p = min(0.95, 0.3 + 0.7 * i / max(1, n_steps - 1))
                scores_seq.append(1.0 if rng_bkt.random() < p else 0.0)
            else:
                scores_seq.append(rng_bkt.choice([0.0, 0.5, 1.0]))

        mastery_vals = [init_mastery]
        p = init_mastery
        for s in scores_seq:
            p = smooth_update(p, s, lr=lr, transit=transit)
            mastery_vals.append(p)

        base_time = datetime(2025, 1, 1)
        conf_vals = []
        recent_window: list[float] = []
        for i, s in enumerate(scores_seq):
            recent_window.append(s)
            if len(recent_window) > 5:
                recent_window.pop(0)
            ra = sum(recent_window) / len(recent_window)
            c = compute_confidence(
                attempts_count=i + 1,
                recent_accuracy=ra,
                probability_effective=mastery_vals[i + 1],
                last_practiced=base_time + timedelta(hours=i),
                now=base_time + timedelta(hours=i + 1),
            )
            conf_vals.append(c)

        steps_x = list(range(1, n_steps + 1))

        st.subheader("Mastery + Confidence")
        fig_bkt = go.Figure()
        fig_bkt.add_trace(go.Scatter(
            x=[0] + steps_x, y=mastery_vals,
            name="Mastery (оценка системы)", line=dict(color="#636EFA", width=3),
        ))
        fig_bkt.add_trace(go.Scatter(
            x=steps_x, y=conf_vals,
            name="Confidence (уверенность)", line=dict(color="#EF553B", width=2, dash="dash"),
        ))
        fig_bkt.add_trace(go.Bar(
            x=steps_x, y=scores_seq,
            name="Балл за попытку", marker_color="rgba(99,110,250,0.15)",
        ))
        fig_bkt.update_layout(
            xaxis_title="Попытка", yaxis_title="Значение",
            yaxis_range=[0, 1.05], height=400, barmode="overlay",
        )
        st.plotly_chart(fig_bkt, use_container_width=True)

        st.markdown(
            "**Как это работает:**  \n"
            "- Правильный ответ (1.0) повышает mastery, неправильный (0.0) понижает  \n"
            "- Скорость зависит от learning rate и бонусов (streak, surprise)  \n"
            "- Confidence растёт с числом попыток и стабильностью  \n"
            "- Если ученик давно не практиковался, confidence падает (recency)"
        )

        col_c1, col_c2, col_c3 = st.columns(3)
        final_stability = 1.0 - abs(
            (sum(recent_window) / len(recent_window)) - mastery_vals[-1]
        )
        col_c1.metric("Финальный Mastery", f"{mastery_vals[-1]:.3f}")
        col_c2.metric("Финальный Confidence", f"{conf_vals[-1]:.3f}")
        col_c3.metric("Stability", f"{final_stability:.3f}")


