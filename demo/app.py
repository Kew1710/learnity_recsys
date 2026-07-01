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
st.title("Learnity RecSys -- адаптивная рекомендательная система")
st.caption("Интерактивная in-memory симуляция. Сервисы и БД не требуются.")

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

tab_full, tab_micro, tab_macro, tab_mastery, tab_cluster = st.tabs([
    "Полная симуляция",
    "Micro-уровень",
    "Macro-уровень",
    "Mastery и Confidence",
    "Кластеризация",
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
            n_peers = 60
            peer_data = np.clip(
                rng_np.rand(n_peers, len(sim_kcs)) * 0.8 + 0.1, 0, 1,
            ).astype(np.float32)
            student_vec = np.array([vis_m[kc] for kc in sim_kcs], dtype=np.float32)
            full_matrix = np.vstack([peer_data, student_vec.reshape(1, -1)])
            gmm, best_k = _select_n_clusters(full_matrix, max_k=8)
            all_labels = gmm.predict(full_matrix)
            student_cluster = int(all_labels[-1])

            st.markdown("**Mastery-вектор ученика (после CAT):**")
            vec_df = pd.DataFrame([{
                KC_NAMES[kc]: round(vis_m[kc], 2) for kc in sim_kcs
            }])
            st.dataframe(vec_df, use_container_width=True, hide_index=True)

            pca = PCA(n_components=2)
            coords = pca.fit_transform(full_matrix)
            labels_str = [f"Кластер {l}" for l in all_labels[:-1]] + ["Наш ученик"]
            sizes = [6] * n_peers + [16]
            fig_cl = px.scatter(
                x=coords[:, 0], y=coords[:, 1],
                color=labels_str, size=sizes,
                labels={"x": "PC1", "y": "PC2", "color": ""},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_cl.update_layout(height=350, margin=dict(t=10, b=10))
            st.plotly_chart(fig_cl, use_container_width=True)

            cluster_size = int((all_labels == student_cluster).sum())
            st.success(
                f"Найдено **{best_k}** кластеров. "
                f"Ученик отнесен к **кластеру {student_cluster}** "
                f"({cluster_size} уч.). "
                f"Thompson Sampling получит prior из этого кластера."
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
                    "Для каждой темы Micro-уровень подбирает задания (IRT-фильтр + ZPD). "
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
                        row = {
                            "N": tasks_done,
                            "Задание": task["task_id"],
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
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Всего заданий", len(history))
                acc = sum(1 for s in scores_all if s >= 0.9) / len(scores_all)
                m2.metric("Точность", f"{acc:.0%}")
                m3.metric("Средний балл", f"{sum(scores_all) / len(scores_all):.2f}")
                m4.metric("Тем пройдено", len(plan))

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
        "Micro-уровень (Retrieval) отвечает за выбор конкретного задания. "
        "Три ключевых компонента: IRT-фильтр, Thompson Sampling, диагностический CAT."
    )

    micro_section = st.radio(
        "Компонент",
        ["IRT Pre-filter", "Thompson Sampling", "Диагностический CAT"],
        horizontal=True,
        key="micro_section",
    )

    if micro_section == "IRT Pre-filter":
        st.subheader("IRT Pre-filter: фильтрация заданий по сложности")
        st.markdown(
            "Система оставляет задания, где вероятность правильного ответа P(correct) "
            "попадает в диапазон [0.20, 0.90]. Слишком лёгкие и слишком сложные отсекаются."
        )

        col_irt_cfg, col_irt_viz = st.columns([1, 2])
        with col_irt_cfg:
            irt_mastery = st.slider("Mastery ученика", 0.05, 0.95, 0.50, 0.05, key="irt_m")
            irt_mode = st.selectbox("Режим обучения", ["build", "consolidate", "test"], key="irt_mode")
            irt_kc = st.selectbox(
                "KC для примера",
                ALL_KCS[:8],
                format_func=lambda x: KC_NAMES[x],
                key="irt_kc",
            )

        with col_irt_viz:
            kc_tasks_demo = [t for t in TASK_POOL if t["kc_id"] == irt_kc]
            filtered_demo, fallback_used = filter_tasks_by_irt(kc_tasks_demo, irt_mastery)

            task_rows = []
            for t in kc_tasks_demo:
                d_val = t["parts"][0]["irt_difficulty"]
                p = compute_p_correct(irt_mastery, d_val)
                passes = t in filtered_demo
                task_rows.append({
                    "Задание": t["task_id"],
                    "Сложность": diff_label(d_val),
                    "IRT diff": round(d_val, 2),
                    "P(correct)": round(p, 3),
                    "Статус": ("Fallback" if fallback_used else "Принято") if passes else "Отклонено",
                })
            st.dataframe(pd.DataFrame(task_rows), use_container_width=True, hide_index=True)

            if fallback_used:
                st.warning("Ни одно задание не попало в целевой диапазон. Выбраны ближайшие (fallback).")

            theta_irt = math.log(max(0.01, min(0.99, irt_mastery)) / (1 - max(0.01, min(0.99, irt_mastery))))
            diffs_range = np.linspace(-3.0, 3.0, 200)
            p_curve = [1.0 / (1.0 + math.exp(-(theta_irt - d))) for d in diffs_range]
            fig_irt = go.Figure()
            fig_irt.add_trace(go.Scatter(
                x=diffs_range, y=p_curve, name="P(correct)",
                line=dict(color="#636EFA", width=2),
            ))
            fig_irt.add_hline(y=0.20, line_dash="dash", line_color="red",
                              annotation_text="floor (0.20)")
            fig_irt.add_hline(y=0.90, line_dash="dash", line_color="red",
                              annotation_text="ceiling (0.90)")
            for t in kc_tasks_demo:
                d_val = t["parts"][0]["irt_difficulty"]
                p = compute_p_correct(irt_mastery, d_val)
                color = "green" if t in filtered_demo else "red"
                fig_irt.add_trace(go.Scatter(
                    x=[d_val], y=[p], mode="markers",
                    marker=dict(size=10, color=color),
                    name=t["task_id"], showlegend=False,
                ))
            fig_irt.update_layout(
                xaxis_title="Сложность задания (IRT logit-шкала)",
                yaxis_title="P(correct)",
                yaxis_range=[-0.05, 1.05], height=400,
            )
            st.plotly_chart(fig_irt, use_container_width=True)

    elif micro_section == "Thompson Sampling":
        st.subheader("Thompson Sampling: байесовский выбор задания")
        st.markdown(
            "Бандит оценивает ожидаемую награду для каждого задания. "
            "Thompson Sampling сэмплирует из апостериорного распределения -- "
            "это даёт баланс между использованием лучшего и проверкой неизвестных."
        )

        col_ts_cfg, col_ts_viz = st.columns([1, 3])
        with col_ts_cfg:
            n_arms = st.slider("Число заданий (arms)", 3, 8, 5, key="ts_arms")
            n_rounds = st.slider("Число раундов", 20, 300, 100, key="ts_rounds")
            run_ts = st.button("Запустить", type="primary", key="ts_run")

        with col_ts_viz:
            if not run_ts:
                st.info("Нажмите **Запустить** для начала.")
            else:
                rng_ts = np.random.RandomState(42)
                arm_features = rng_ts.randn(n_arms, CONTEXT_DIM).astype(np.float64)
                true_theta = rng_ts.randn(CONTEXT_DIM).astype(np.float64)
                true_theta /= np.linalg.norm(true_theta)
                true_probs = 1.0 / (1.0 + np.exp(-(arm_features @ true_theta)))
                best_arm = int(np.argmax(true_probs))

                ts_model = ThompsonModel.init("demo_kc", cluster_id=0)
                ts_cumrew, rand_cumrew = [], []
                ts_total, rand_total = 0.0, 0.0
                ts_sel = np.zeros(n_arms)

                for t in range(n_rounds):
                    ts_scores = [ts_model.score(arm_features[a]) for a in range(n_arms)]
                    ts_choice = int(np.argmax(ts_scores))
                    ts_reward = float(rng_ts.binomial(1, true_probs[ts_choice]))
                    ts_model.update(arm_features[ts_choice], ts_reward)
                    ts_total += ts_reward
                    ts_cumrew.append(ts_total)
                    ts_sel[ts_choice] += 1

                    rand_choice = rng_ts.randint(n_arms)
                    rand_reward = float(rng_ts.binomial(1, true_probs[rand_choice]))
                    rand_total += rand_reward
                    rand_cumrew.append(rand_total)

                rounds_x = list(range(1, n_rounds + 1))
                oracle = [true_probs[best_arm] * t for t in rounds_x]

                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(
                    x=rounds_x, y=ts_cumrew, name="Thompson Sampling",
                    line=dict(color="#636EFA", width=2),
                ))
                fig_ts.add_trace(go.Scatter(
                    x=rounds_x, y=rand_cumrew, name="Случайный выбор",
                    line=dict(color="#EF553B", width=2),
                ))
                fig_ts.add_trace(go.Scatter(
                    x=rounds_x, y=oracle, name="Oracle (лучший arm)",
                    line=dict(color="#00CC96", width=1, dash="dot"),
                ))
                fig_ts.update_layout(
                    xaxis_title="Раунд", yaxis_title="Кумулятивная награда",
                    height=350,
                )
                st.plotly_chart(fig_ts, use_container_width=True)

                arm_labels = [f"Задание {i} (p={true_probs[i]:.2f})" for i in range(n_arms)]
                fig_sel = px.bar(x=arm_labels, y=ts_sel, color=ts_sel,
                                 color_continuous_scale="Blues")
                fig_sel.update_layout(
                    height=280, showlegend=False, coloraxis_showscale=False,
                    xaxis_title="", yaxis_title="Число выборов",
                    title="Частота выбора каждого задания",
                )
                st.plotly_chart(fig_sel, use_container_width=True)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Награда TS", f"{ts_cumrew[-1]:.0f}")
                mc2.metric("Награда Random", f"{rand_cumrew[-1]:.0f}")
                mc3.metric("Regret TS", f"{oracle[-1] - ts_cumrew[-1]:.0f}")

    else:  # Диагностический CAT
        st.subheader("Диагностический CAT: калибровка нового ученика")
        st.markdown(
            "Для нового ученика система проводит 5-8 адаптивных заданий. "
            "Каждое задание выбирается так, чтобы максимизировать информацию Фишера: "
            "I(theta) = P(correct) * (1 - P(correct)), максимум при P = 0.5."
        )

        col_cat_cfg, col_cat_viz = st.columns([1, 3])
        with col_cat_cfg:
            cat_grade = st.slider("Класс", 1, 11, 7, key="cat_grade")
            cat_type = st.selectbox("Тип ученика", list(STUDENT_TYPES.keys()), key="cat_type")
            run_cat = st.button("Запустить CAT", type="primary", key="cat_run")

        with col_cat_viz:
            if not run_cat:
                theta_demo = st.slider("theta ученика", -3.0, 3.0, 0.0, 0.1, key="cat_theta")
                diffs_cat = np.linspace(-3, 3, 200)
                p_vals = 1.0 / (1.0 + np.exp(-(theta_demo - diffs_cat)))
                fisher_vals = p_vals * (1.0 - p_vals)

                fig_fi = go.Figure()
                fig_fi.add_trace(go.Scatter(
                    x=diffs_cat, y=fisher_vals, name="I(theta)",
                    line=dict(color="#636EFA", width=2),
                ))
                fig_fi.add_trace(go.Scatter(
                    x=diffs_cat, y=p_vals, name="P(correct)",
                    line=dict(color="#EF553B", width=1, dash="dash"),
                ))
                fig_fi.add_vline(x=theta_demo, line_dash="dot", line_color="green",
                                 annotation_text=f"theta={theta_demo:.1f}")
                fig_fi.update_layout(
                    xaxis_title="Сложность задания",
                    yaxis_title="Значение", height=350,
                    title="Информация Фишера: максимум при difficulty = theta",
                )
                st.plotly_chart(fig_fi, use_container_width=True)
                st.caption(
                    "Двигайте ползунок theta, чтобы увидеть, как смещается пик "
                    "информации Фишера. Система выбирает задание с difficulty "
                    "рядом с текущей оценкой theta."
                )
            else:
                cat_kcs = grade_kcs(cat_grade)
                cat_true_m = init_true_mastery(cat_grade, STUDENT_TYPES[cat_type]["mod"])
                cat_ps = STUDENT_TYPES[cat_type]["p_slip"]
                cat_pg = STUDENT_TYPES[cat_type]["p_guess"]
                cat_rng = random.Random(7)
                cat_st = CATState.from_mastery({kc: 0.50 for kc in cat_kcs})
                cat_steps_log: list[dict] = []

                while not cat_st.is_complete:
                    kc = select_diagnostic_kc(cat_st, cat_kcs)
                    if kc is None:
                        break
                    kc_tasks_cat = [t for t in TASK_POOL if t["kc_id"] == kc]
                    task_cat = select_diagnostic_task(
                        kc_tasks_cat, cat_st.kc_theta[kc],
                    )
                    if task_cat is None:
                        break
                    diff_val = task_cat["parts"][0]["irt_difficulty"]
                    p_before = 1.0 / (1.0 + math.exp(-cat_st.kc_theta[kc]))
                    sc = simulate_answer(
                        cat_true_m[kc], diff_val, cat_ps, cat_pg, cat_rng,
                    )
                    update_cat_state(cat_st, kc, sc, diff_val)
                    p_after = 1.0 / (1.0 + math.exp(-cat_st.kc_theta[kc]))
                    cat_steps_log.append({
                        "N": len(cat_steps_log) + 1,
                        "Тема": KC_NAMES[kc],
                        "Сложность": diff_label(diff_val),
                        "Ответ": score_label(sc),
                        "Mastery до": round(p_before, 3),
                        "Mastery после": round(p_after, 3),
                        "Истинный": round(cat_true_m[kc], 3),
                    })

                st.dataframe(
                    pd.DataFrame(cat_steps_log),
                    use_container_width=True, hide_index=True,
                )
                cat_final_m = cat_st.to_mastery()
                kc_labels = [KC_NAMES[kc] for kc in cat_kcs]
                fig_cat_comp = go.Figure()
                fig_cat_comp.add_trace(go.Bar(
                    x=kc_labels, y=[0.50] * len(cat_kcs),
                    name="Prior (0.50)", marker_color="lightgray",
                ))
                fig_cat_comp.add_trace(go.Bar(
                    x=kc_labels, y=[cat_final_m[kc] for kc in cat_kcs],
                    name="После CAT", marker_color="#636EFA",
                ))
                fig_cat_comp.add_trace(go.Bar(
                    x=kc_labels, y=[cat_true_m[kc] for kc in cat_kcs],
                    name="Истинный", marker_color="#00CC96", opacity=0.5,
                ))
                fig_cat_comp.update_layout(
                    barmode="group", height=400,
                    yaxis_title="Mastery", yaxis_range=[0, 1.05],
                    title=f"Калибровка за {len(cat_steps_log)} заданий",
                )
                st.plotly_chart(fig_cat_comp, use_container_width=True)


# ===================================================================
# TAB 3 -- Macro-уровень
# ===================================================================
with tab_macro:
    st.header("Macro-уровень: управление планом обучения")
    st.markdown(
        "Macro-уровень анализирует прогресс ученика и принимает стратегические решения: "
        "продвинуть по плану, вставить дополнительную тему или перейти в режим закрепления."
    )

    macro_section = st.radio(
        "Компонент",
        ["Diagnostic Layer", "MicroSummary -> Решение"],
        horizontal=True,
        key="macro_section",
    )

    if macro_section == "Diagnostic Layer":
        st.subheader("Diagnostic Layer: анализ причин затруднений")
        st.markdown(
            "Система определяет **причину** проблемы, а не просто реагирует на симптомы. "
            "4 типа диагнозов + on_track для нормального прогресса."
        )

        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            d_mastery = st.slider("Текущий mastery", 0.0, 1.0, 0.40, 0.05, key="d_m")
            d_velocity = st.slider("Velocity", -0.10, 0.10, 0.00, 0.01, key="d_v")
            d_frust = st.slider("Frustration (ошибки подряд)", 0, 5, 2, key="d_f")
            d_score = st.slider("Средний балл", 0.0, 1.0, 0.40, 0.05, key="d_s")
        with col_d2:
            d_tasks = st.slider("Заданий на шаге", 1, 20, 5, key="d_t")
            d_attempts = st.slider("Всего попыток по KC", 1, 30, 8, key="d_a")
            d_conf = st.slider("Confidence", 0.0, 1.0, 0.50, 0.05, key="d_c")
            d_prereq = st.slider("Mastery пререквизита", 0.0, 1.0, 0.60, 0.05, key="d_p")
            d_tc = st.slider("Заданий в банке", 1, 20, 10, key="d_tc")

        d_result = diagnose(
            mastery_current=d_mastery, velocity=d_velocity,
            frustration_count=d_frust, avg_score=d_score,
            tasks_spent=d_tasks, attempts_count=d_attempts,
            mastery_confidence=d_conf, weakest_prereq_mastery=d_prereq,
            task_count_for_kc=d_tc,
        )
        label, desc = DIAGNOSIS_RU[d_result.reason]
        color_map = {
            "prereq_gap": "red", "content_gap": "orange",
            "uncertain_estimate": "blue", "regression": "violet",
            "on_track": "green",
        }
        st.markdown(f"### Результат: :{color_map[d_result.reason]}[{label}]")
        st.markdown(f"**Уверенность:** {d_result.confidence:.0%}")
        st.markdown(f"**Описание:** {desc}")
        st.markdown(f"**Детали (из модели):** {d_result.detail}")

        st.divider()
        st.markdown("#### Примеры всех диагнозов:")
        examples = [
            ("on_track", dict(mastery_current=0.5, velocity=0.03, frustration_count=0,
                              avg_score=0.7, tasks_spent=3, attempts_count=8,
                              mastery_confidence=0.6, weakest_prereq_mastery=0.8,
                              task_count_for_kc=10)),
            ("prereq_gap", dict(mastery_current=0.3, velocity=-0.02, frustration_count=3,
                                avg_score=0.3, tasks_spent=5, attempts_count=8,
                                mastery_confidence=0.6, weakest_prereq_mastery=0.35,
                                task_count_for_kc=10)),
            ("uncertain_estimate", dict(mastery_current=0.4, velocity=0.0, frustration_count=2,
                                        avg_score=0.4, tasks_spent=2, attempts_count=3,
                                        mastery_confidence=0.2, weakest_prereq_mastery=0.7,
                                        task_count_for_kc=10)),
            ("regression", dict(mastery_current=0.8, velocity=-0.05, frustration_count=3,
                                avg_score=0.3, tasks_spent=4, attempts_count=15,
                                mastery_confidence=0.7, weakest_prereq_mastery=0.8,
                                task_count_for_kc=10)),
            ("content_gap", dict(mastery_current=0.4, velocity=-0.01, frustration_count=3,
                                 avg_score=0.35, tasks_spent=6, attempts_count=10,
                                 mastery_confidence=0.5, weakest_prereq_mastery=0.7,
                                 task_count_for_kc=2)),
        ]
        for _expected, params in examples:
            diag = diagnose(**params)
            lbl, dsc = DIAGNOSIS_RU[diag.reason]
            clr = color_map[diag.reason]
            st.markdown(f":{clr}[**{lbl}**] ({diag.confidence:.0%}) -- {dsc}")

    else:  # MicroSummary -> Решение
        st.subheader("MicroSummary -> Macro-решение")
        st.markdown(
            "Micro-уровень собирает статистику по последним заданиям. "
            "Macro-уровень анализирует её и принимает решение."
        )

        score_input = st.text_input(
            "Последовательность ответов (через запятую: 0 / 0.5 / 1)",
            "1, 1, 0.5, 0, 0, 1, 0, 0, 0, 1",
            key="ms_input",
        )
        ms_mastery = st.slider("Текущий mastery", 0.0, 1.0, 0.45, 0.05, key="ms_m")
        ms_prereq = st.slider("Mastery пререквизита", 0.0, 1.0, 0.65, 0.05, key="ms_prereq")

        try:
            scores_list = [float(s.strip()) for s in score_input.split(",") if s.strip()]
        except ValueError:
            scores_list = []
            st.error("Некорректный ввод.")

        if scores_list:
            mock_history = [{
                "kc_id": "demo_kc", "score": sc,
                "irt_diff": -0.5 + 0.1 * i, "mastery_delta": 0.02 * (sc - 0.5),
            } for i, sc in enumerate(scores_list)]

            ms = compute_micro_summary(mock_history, "demo_kc", ms_mastery)

            col_ms1, col_ms2, col_ms3, col_ms4 = st.columns(4)
            col_ms1.metric("Средний балл", f"{ms['avg_score']:.2f}")
            col_ms2.metric("Velocity", f"{ms['velocity']:+.4f}")
            col_ms3.metric("Frustration", ms["frustration_count"])
            col_ms4.metric("IRT residual", f"{ms['irt_residual']:.3f}")

            conf = compute_confidence(
                attempts_count=len(scores_list),
                recent_accuracy=ms["avg_score"],
                probability_effective=ms_mastery,
                last_practiced=datetime(2025, 1, 1),
                now=datetime(2025, 1, 1, 0, 1),
            )

            d_ms = diagnose(
                mastery_current=ms_mastery, velocity=ms["velocity"],
                frustration_count=ms["frustration_count"],
                avg_score=ms["avg_score"],
                tasks_spent=len(scores_list), attempts_count=len(scores_list),
                mastery_confidence=conf, weakest_prereq_mastery=ms_prereq,
                task_count_for_kc=10,
            )
            lbl_ms, desc_ms = DIAGNOSIS_RU[d_ms.reason]
            clr_ms = {"prereq_gap": "red", "content_gap": "orange",
                       "uncertain_estimate": "blue", "regression": "violet",
                       "on_track": "green"}[d_ms.reason]

            st.markdown("---")
            st.markdown("#### Цепочка принятия решения:")
            st.info(
                f"**[MICRO -> MicroSummary]** avg_score={ms['avg_score']:.2f}, "
                f"velocity={ms['velocity']:+.4f}, frustration={ms['frustration_count']}"
            )
            st.markdown(f"**[MACRO -> Диагностика]** :{clr_ms}[{lbl_ms}] ({d_ms.confidence:.0%})")

            if ms_mastery >= 0.75:
                st.success("**[MACRO -> Решение]** Mastery достаточен -> ADVANCE")
            elif d_ms.reason == "prereq_gap":
                st.warning("**[MACRO -> Решение]** INSERT_PREREQ (вставить пререквизит)")
            elif ms["frustration_count"] >= 3:
                st.warning("**[MACRO -> Решение]** CONSOLIDATE (режим закрепления)")
            elif d_ms.reason == "on_track":
                st.info("**[MACRO -> Решение]** CONTINUE (продолжить обучение)")
            else:
                st.info("**[MACRO -> Решение]** CONTINUE (наблюдать)")

            fig_scores = go.Figure()
            fig_scores.add_trace(go.Bar(
                x=list(range(1, len(scores_list) + 1)), y=scores_list,
                marker_color=[
                    "#00CC96" if s >= 0.9 else "#FFA15A" if s >= 0.4 else "#EF553B"
                    for s in scores_list
                ],
            ))
            fig_scores.update_layout(
                xaxis_title="Задание", yaxis_title="Балл",
                yaxis_range=[0, 1.1], height=250,
                title="Последовательность ответов",
            )
            st.plotly_chart(fig_scores, use_container_width=True)


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


# ===================================================================
# TAB 5 -- Кластеризация
# ===================================================================
with tab_cluster:
    st.header("Кластеризация учеников (GMM + BIC)")
    st.markdown(
        "Система группирует учеников по профилю знаний с помощью Gaussian Mixture Model. "
        "Число кластеров подбирается автоматически через BIC. "
        "Новый ученик получает Thompson Sampling prior из своего кластера."
    )

    col_cl_cfg, col_cl_viz = st.columns([1, 3])

    with col_cl_cfg:
        st.subheader("Параметры")
        n_students = st.slider("Число учеников", 30, 300, 100, key="cl_n")
        n_kc_dim = st.slider("Число KC (размерность)", 5, 15, 8, key="cl_dim")
        true_k = st.slider("Истинное число кластеров", 2, 8, 4, key="cl_k")
        max_k_search = st.slider("Макс. k для поиска BIC", 3, 15, 10, key="cl_maxk")
        run_cluster = st.button("Запустить кластеризацию", key="cl_run", type="primary")

    with col_cl_viz:
        if not run_cluster:
            st.info("Нажмите **Запустить кластеризацию** для начала.")
        else:
            rng_cl = np.random.RandomState(42)
            centers = rng_cl.rand(true_k, n_kc_dim).astype(np.float32)
            labels_true = rng_cl.randint(0, true_k, size=n_students)
            matrix = np.zeros((n_students, n_kc_dim), dtype=np.float32)
            for i in range(n_students):
                matrix[i] = centers[labels_true[i]] + rng_cl.randn(n_kc_dim).astype(np.float32) * 0.12
            matrix = np.clip(matrix, 0, 1)

            best_gmm, best_k_found = _select_n_clusters(matrix, max_k=max_k_search)
            labels_pred = best_gmm.predict(matrix)

            st.success(f"BIC выбрал k = **{best_k_found}** (истинное k = {true_k})")

            st.subheader("BIC для разных k")
            ks = list(range(2, max_k_search + 1))
            bics = []
            for k in ks:
                gmm_tmp = GaussianMixture(
                    n_components=k, covariance_type="diag", n_init=3, random_state=42,
                )
                gmm_tmp.fit(matrix)
                bics.append(gmm_tmp.bic(matrix))

            fig_bic = go.Figure()
            fig_bic.add_trace(go.Scatter(
                x=ks, y=bics, mode="lines+markers",
                line=dict(color="#636EFA", width=2), marker=dict(size=8),
            ))
            fig_bic.add_vline(
                x=best_k_found, line_dash="dash", line_color="red",
                annotation_text=f"best k={best_k_found}",
            )
            fig_bic.update_layout(
                xaxis_title="k (число кластеров)",
                yaxis_title="BIC (ниже = лучше)", height=350,
            )
            st.plotly_chart(fig_bic, use_container_width=True)

            st.subheader("PCA-проекция (2D)")
            pca_cl = PCA(n_components=2)
            coords_cl = pca_cl.fit_transform(matrix)
            fig_pca = px.scatter(
                x=coords_cl[:, 0], y=coords_cl[:, 1],
                color=[f"Кластер {l}" for l in labels_pred],
                labels={"x": "PC1", "y": "PC2", "color": "Кластер"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pca.update_layout(height=400)
            st.plotly_chart(fig_pca, use_container_width=True)

            st.subheader("Размеры кластеров")
            unique_cl, counts_cl = np.unique(labels_pred, return_counts=True)
            fig_sizes = px.bar(
                x=[f"Кластер {u}" for u in unique_cl], y=counts_cl,
                color=counts_cl, color_continuous_scale="Viridis",
            )
            fig_sizes.update_layout(
                height=280, showlegend=False, coloraxis_showscale=False,
                xaxis_title="", yaxis_title="Число учеников",
            )
            st.plotly_chart(fig_sizes, use_container_width=True)
