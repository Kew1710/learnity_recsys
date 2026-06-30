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
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# Добавляем корень проекта в sys.path чтобы импортировать модули сервисов.
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.profile.bkt import smooth_update, compute_confidence, apply_decay
from services.retrieval.thompson import ThompsonModel
from services.retrieval.linucb import LinUCBModel
from services.retrieval.diagnostic_cat import (
    CATState,
    select_diagnostic_kc,
    select_diagnostic_task,
    update_cat_state,
)
from services.macro.diagnostics import diagnose, Diagnosis
from services.clustering.cluster import _select_n_clusters
from shared.config import retrieval as _rcfg, bkt as _bcfg

CONTEXT_DIM = _rcfg.CONTEXT_DIM

# ---------------------------------------------------------------------------
# Общие настройки страницы
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Learnity RecSys Demo",
    page_icon=":books:",
    layout="wide",
)

st.title("Learnity RecSys -- демонстрация адаптивной системы рекомендаций")
st.caption("Интерактивная in-memory симуляция без запущенных сервисов")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Симуляция обучения",
    "Mastery и Confidence",
    "Thompson Sampling vs LinUCB",
    "Кластеризация",
    "Архитектура системы",
])

# ===================================================================
# Вспомогательные функции
# ===================================================================

KC_POOL = [
    "kc_arithmetic", "kc_fractions", "kc_decimals", "kc_percentages",
    "kc_linear_eq", "kc_quadratic_eq", "kc_geometry_basic",
    "kc_area_perimeter", "kc_angles", "kc_probability",
    "kc_statistics", "kc_functions", "kc_graphs",
    "kc_powers", "kc_roots",
]

RECOMMENDATION_SOURCES = [
    "thompson_sampling", "exploration", "diagnostic_cat", "zpd", "stretch",
]


def simulate_answer_irt(
    true_mastery: float,
    irt_difficulty: float,
    p_slip: float = 0.10,
    p_guess: float = 0.05,
    rng: random.Random | None = None,
) -> float:
    """3-PL IRT ответ студента."""
    rng = rng or random.Random()
    m = max(0.001, min(0.999, true_mastery))
    theta = math.log(m / (1.0 - m))
    p_irt = 1.0 / (1.0 + math.exp(-(theta - irt_difficulty)))
    p_correct = p_guess + (1.0 - p_guess - p_slip) * p_irt
    r = rng.random()
    if r < p_correct:
        return 1.0
    elif r < p_correct + 0.15 * (1.0 - p_correct):
        return 0.5
    return 0.0


def make_task_pool(kc_ids: list[str], n_per_kc: int = 3) -> list[dict]:
    """Генерирует синтетический банк заданий."""
    tasks = []
    rng = random.Random(0)
    for kc in kc_ids:
        for i in range(n_per_kc):
            diff = max(0.05, min(0.95, 0.2 + 0.6 * i / max(1, n_per_kc - 1) + rng.gauss(0, 0.05)))
            tasks.append({
                "task_id": f"{kc}_t{i}",
                "kc_id": kc,
                "parts": [{"irt_difficulty": round(diff, 3)}],
            })
    return tasks


# ===================================================================
# TAB 1 -- Симуляция обучения
# ===================================================================
with tab1:
    st.header("Симуляция учебной сессии")
    st.markdown(
        "Пошаговая симуляция: система рекомендует задание, "
        "ученик отвечает (симулируется через IRT), mastery обновляется."
    )

    col_cfg, col_viz = st.columns([1, 3])

    with col_cfg:
        st.subheader("Параметры")
        grade = st.slider("Класс", 1, 11, 8, key="sim_grade")
        n_tasks = st.slider("Количество заданий", 10, 50, 25, key="sim_n_tasks")
        mode = st.selectbox("Режим обучения", ["build", "consolidate", "test"], key="sim_mode")
        student_type = st.selectbox(
            "Тип ученика",
            ["Средний", "Быстрый", "Медленный", "Продвинутый"],
            key="sim_student",
        )
        run_sim = st.button("Запустить симуляцию", key="sim_run", type="primary")

    with col_viz:
        if run_sim:
            student_params = {
                "Средний":      {"init_mastery": 0.10, "growth": 0.06, "p_slip": 0.12},
                "Быстрый":      {"init_mastery": 0.08, "growth": 0.10, "p_slip": 0.06},
                "Медленный":    {"init_mastery": 0.05, "growth": 0.035, "p_slip": 0.22},
                "Продвинутый":  {"init_mastery": 0.60, "growth": 0.07, "p_slip": 0.05},
            }[student_type]

            n_kcs = min(5 + grade, len(KC_POOL))
            kc_ids = KC_POOL[:n_kcs]
            task_pool = make_task_pool(kc_ids)
            rng = random.Random(42)

            true_mastery = {kc: student_params["init_mastery"] for kc in kc_ids}
            visible_mastery = {kc: 0.50 for kc in kc_ids}

            mode_offsets = {"build": 0.10, "consolidate": -0.10, "test": 0.30}
            offset = mode_offsets[mode]

            history: list[dict] = []
            source_counts: dict[str, int] = {s: 0 for s in RECOMMENDATION_SOURCES}

            progress_bar = st.progress(0, text="Симуляция...")

            for step in range(1, n_tasks + 1):
                # Выбираем KC: ту, у которой наибольший потенциал (разрыв mastery - target)
                target_mastery = {kc: max(0.05, min(0.95, visible_mastery[kc] + offset)) for kc in kc_ids}
                kc_id = min(kc_ids, key=lambda k: visible_mastery[k] - target_mastery[k] + rng.gauss(0, 0.05))

                # Рекомендуем задание
                kc_tasks = [t for t in task_pool if t["kc_id"] == kc_id]
                target_diff = max(0.05, min(0.95, visible_mastery[kc_id] + offset))

                # Стохастический выбор источника
                r = rng.random()
                if step <= 3:
                    source = "diagnostic_cat"
                elif r < 0.10:
                    source = "exploration"
                elif r < 0.20:
                    source = "stretch"
                else:
                    source = "thompson_sampling"
                source_counts[source] = source_counts.get(source, 0) + 1

                if source == "exploration":
                    task = rng.choice(kc_tasks)
                else:
                    task = min(kc_tasks, key=lambda t: abs(t["parts"][0]["irt_difficulty"] - target_diff))

                irt_diff = task["parts"][0]["irt_difficulty"]
                score = simulate_answer_irt(
                    true_mastery[kc_id], irt_diff,
                    p_slip=student_params["p_slip"], rng=rng,
                )

                # Обновляем visible mastery через smooth_update из проекта
                visible_mastery[kc_id] = smooth_update(visible_mastery[kc_id], score)

                # Обновляем true mastery (скрытое от системы)
                true_mastery[kc_id] = min(
                    1.0,
                    true_mastery[kc_id] + student_params["growth"] * score * (1.0 - true_mastery[kc_id]),
                )

                history.append({
                    "step": step,
                    "kc_id": kc_id,
                    "irt_difficulty": irt_diff,
                    "score": score,
                    "visible_mastery": visible_mastery[kc_id],
                    "true_mastery": true_mastery[kc_id],
                    "source": source,
                })
                progress_bar.progress(step / n_tasks, text=f"Шаг {step}/{n_tasks}")

            progress_bar.empty()

            # --- Метрики ---
            scores = [h["score"] for h in history]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Точность", f"{sum(1 for s in scores if s >= 0.9) / len(scores):.0%}")
            m2.metric("Средний балл", f"{sum(scores) / len(scores):.2f}")
            mae = np.mean([abs(history[-1]["true_mastery"] - history[-1]["visible_mastery"])])
            m3.metric("MAE (true vs vis)", f"{mae:.3f}")
            m4.metric("KC задействовано", f"{len({h['kc_id'] for h in history})}")

            # --- Mastery evolution ---
            st.subheader("Эволюция mastery по шагам")

            kc_traces_vis: dict[str, list] = {kc: [] for kc in kc_ids}
            kc_traces_true: dict[str, list] = {kc: [] for kc in kc_ids}
            kc_steps: dict[str, list] = {kc: [] for kc in kc_ids}

            vis_running = {kc: 0.50 for kc in kc_ids}
            true_running = {kc: student_params["init_mastery"] for kc in kc_ids}
            for h in history:
                kc = h["kc_id"]
                vis_running[kc] = h["visible_mastery"]
                true_running[kc] = h["true_mastery"]
                kc_traces_vis[kc].append(vis_running[kc])
                kc_traces_true[kc].append(true_running[kc])
                kc_steps[kc].append(h["step"])

            fig_mastery = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, kc in enumerate(kc_ids):
                if not kc_steps[kc]:
                    continue
                c = colors[i % len(colors)]
                fig_mastery.add_trace(go.Scatter(
                    x=kc_steps[kc], y=kc_traces_vis[kc],
                    name=f"{kc} (visible)", line=dict(color=c),
                ))
                fig_mastery.add_trace(go.Scatter(
                    x=kc_steps[kc], y=kc_traces_true[kc],
                    name=f"{kc} (true)", line=dict(color=c, dash="dot"),
                ))
            fig_mastery.update_layout(
                xaxis_title="Шаг", yaxis_title="Mastery",
                yaxis_range=[0, 1], height=400,
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_mastery, use_container_width=True)

            # --- Источники рекомендаций ---
            col_pie, col_log = st.columns([1, 2])
            with col_pie:
                st.subheader("Источники рекомендаций")
                non_zero = {k: v for k, v in source_counts.items() if v > 0}
                fig_pie = px.pie(
                    names=list(non_zero.keys()),
                    values=list(non_zero.values()),
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_pie.update_layout(height=300, margin=dict(t=20, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_log:
                st.subheader("Последние шаги")
                import pandas as pd
                df_log = pd.DataFrame(history[-10:])
                df_log["score"] = df_log["score"].map({1.0: "1.0", 0.5: "0.5", 0.0: "0.0"})
                for col in ["visible_mastery", "true_mastery", "irt_difficulty"]:
                    df_log[col] = df_log[col].round(3)
                st.dataframe(df_log, use_container_width=True, hide_index=True)
        else:
            st.info("Нажмите **Запустить симуляцию** для начала.")


# ===================================================================
# TAB 2 -- Mastery и Confidence
# ===================================================================
with tab2:
    st.header("Модель Mastery: BKT smooth_update")
    st.markdown(
        "Формула: `new_p = p + lr * (score - p) + transit * (1 - new_p)`  \n"
        "Эквивалентно EMA с убывающей скоростью обновления."
    )

    col_sliders, col_charts = st.columns([1, 2])

    with col_sliders:
        st.subheader("Параметры")
        init_mastery = st.slider("Начальный mastery", 0.0, 1.0, 0.10, 0.01, key="bkt_init")
        lr = st.slider("Learning rate (lr)", 0.01, 0.50, float(_bcfg.SMOOTH_LR), 0.01, key="bkt_lr")
        transit = st.slider("Transit", 0.0, 0.10, float(_bcfg.SMOOTH_TRANSIT), 0.005, key="bkt_transit")
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
        n_steps = st.slider("Количество шагов", 5, 40, 20, key="bkt_steps")

    with col_charts:
        # Генерируем последовательность ответов
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

        # Прогоняем smooth_update
        mastery_vals = [init_mastery]
        p = init_mastery
        for s in scores_seq:
            p = smooth_update(p, s, lr=lr, transit=transit)
            mastery_vals.append(p)

        # Confidence
        from datetime import datetime, timedelta
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

        # Mastery chart
        st.subheader("Mastery + Confidence")
        fig_bkt = go.Figure()
        fig_bkt.add_trace(go.Scatter(
            x=[0] + steps_x, y=mastery_vals,
            name="Mastery", line=dict(color="#636EFA", width=3),
        ))
        fig_bkt.add_trace(go.Scatter(
            x=steps_x, y=conf_vals,
            name="Confidence", line=dict(color="#EF553B", width=2, dash="dash"),
        ))
        fig_bkt.add_trace(go.Bar(
            x=steps_x, y=scores_seq,
            name="Score", marker_color="rgba(99,110,250,0.2)",
            yaxis="y",
        ))
        fig_bkt.update_layout(
            xaxis_title="Попытка", yaxis_title="Значение",
            yaxis_range=[0, 1.05], height=400, barmode="overlay",
        )
        st.plotly_chart(fig_bkt, use_container_width=True)

        # Confidence breakdown
        st.subheader("Разложение Confidence")
        st.markdown(
            "**confidence** = min(1, (attempts / SCALE) * stability * recency)  \n"
            "- **stability** = 1 - |recent_accuracy - mastery|  \n"
            "- **recency** = 0.5^(days_since / half_life)"
        )

        col_c1, col_c2, col_c3 = st.columns(3)
        # Final state
        final_attempts = n_steps
        final_stability = 1.0 - abs(
            (sum(recent_window) / len(recent_window)) - mastery_vals[-1]
        )
        final_recency = 0.5 ** (1.0 / 24 / _bcfg.HALF_LIFE_DAYS)  # 1 hour gap
        col_c1.metric("Attempts / Scale", f"{final_attempts}/{_bcfg.CONFIDENCE_ATTEMPTS_SCALE}")
        col_c2.metric("Stability", f"{final_stability:.3f}")
        col_c3.metric("Recency", f"{final_recency:.3f}")


# ===================================================================
# TAB 3 -- Thompson Sampling vs LinUCB
# ===================================================================
with tab3:
    st.header("Thompson Sampling vs LinUCB")
    st.markdown(
        "Сравнение двух алгоритмов контекстного бандита на одном наборе заданий.  \n"
        "Оба алгоритма используют одинаковые контекстные вектора и обновляют свои модели online."
    )

    col_bandit_cfg, col_bandit_viz = st.columns([1, 3])

    with col_bandit_cfg:
        st.subheader("Параметры")
        n_arms = st.slider("Число заданий (arms)", 3, 10, 6, key="bandit_arms")
        n_rounds = st.slider("Число раундов", 50, 500, 200, key="bandit_rounds")
        bandit_seed = st.number_input("Random seed", value=42, key="bandit_seed")
        run_bandit = st.button("Запустить сравнение", key="bandit_run", type="primary")

    with col_bandit_viz:
        if run_bandit:
            rng_b = np.random.RandomState(int(bandit_seed))

            # Генерируем контекстные вектора для заданий
            arm_features = rng_b.randn(n_arms, CONTEXT_DIM).astype(np.float64)
            # "Истинный" вектор весов для генерации наград
            true_theta = rng_b.randn(CONTEXT_DIM).astype(np.float64)
            true_theta /= np.linalg.norm(true_theta)

            # Истинная ожидаемая награда для каждого arm
            true_rewards = arm_features @ true_theta
            # Нормализуем в [0, 1] через sigmoid
            true_probs = 1.0 / (1.0 + np.exp(-true_rewards))
            best_arm = int(np.argmax(true_probs))

            ts_model = ThompsonModel.init("demo_kc", cluster_id=0)
            lu_model = LinUCBModel.init("demo_kc", cluster_id=0)

            ts_cumrew = []
            lu_cumrew = []
            ts_selections = np.zeros(n_arms)
            lu_selections = np.zeros(n_arms)
            ts_total = 0.0
            lu_total = 0.0

            for t in range(n_rounds):
                # Thompson Sampling
                ts_scores = [ts_model.score(arm_features[a]) for a in range(n_arms)]
                ts_choice = int(np.argmax(ts_scores))
                ts_reward = float(rng_b.binomial(1, true_probs[ts_choice]))
                ts_model.update(arm_features[ts_choice], ts_reward)
                ts_total += ts_reward
                ts_cumrew.append(ts_total)
                ts_selections[ts_choice] += 1

                # LinUCB
                lu_scores = [lu_model.score(arm_features[a]) for a in range(n_arms)]
                lu_choice = int(np.argmax(lu_scores))
                lu_reward = float(rng_b.binomial(1, true_probs[lu_choice]))
                lu_model.update(arm_features[lu_choice], lu_reward)
                lu_total += lu_reward
                lu_cumrew.append(lu_total)
                lu_selections[lu_choice] += 1

            # --- Cumulative reward ---
            st.subheader("Кумулятивная награда")
            rounds_x = list(range(1, n_rounds + 1))
            oracle_rew = [true_probs[best_arm] * t for t in rounds_x]

            fig_rew = go.Figure()
            fig_rew.add_trace(go.Scatter(
                x=rounds_x, y=ts_cumrew,
                name="Thompson Sampling", line=dict(color="#636EFA", width=2),
            ))
            fig_rew.add_trace(go.Scatter(
                x=rounds_x, y=lu_cumrew,
                name="LinUCB", line=dict(color="#EF553B", width=2),
            ))
            fig_rew.add_trace(go.Scatter(
                x=rounds_x, y=oracle_rew,
                name="Oracle (best arm)", line=dict(color="#00CC96", width=1, dash="dot"),
            ))
            fig_rew.update_layout(
                xaxis_title="Раунд", yaxis_title="Кумулятивная награда",
                height=400,
            )
            st.plotly_chart(fig_rew, use_container_width=True)

            # --- Selection patterns ---
            st.subheader("Частота выбора заданий")
            col_ts, col_lu = st.columns(2)
            arm_labels = [f"Задание {i} (p={true_probs[i]:.2f})" for i in range(n_arms)]

            with col_ts:
                fig_ts_sel = px.bar(
                    x=arm_labels, y=ts_selections,
                    title="Thompson Sampling",
                    color=ts_selections,
                    color_continuous_scale="Blues",
                )
                fig_ts_sel.update_layout(
                    height=300, showlegend=False,
                    xaxis_title="", yaxis_title="Выборов",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_ts_sel, use_container_width=True)

            with col_lu:
                fig_lu_sel = px.bar(
                    x=arm_labels, y=lu_selections,
                    title="LinUCB",
                    color=lu_selections,
                    color_continuous_scale="Reds",
                )
                fig_lu_sel.update_layout(
                    height=300, showlegend=False,
                    xaxis_title="", yaxis_title="Выборов",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_lu_sel, use_container_width=True)

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Суммарная награда (TS)",
                f"{ts_cumrew[-1]:.0f}",
                delta=f"{ts_cumrew[-1] - lu_cumrew[-1]:+.0f} vs LinUCB",
            )
            m2.metric("Суммарная награда (LinUCB)", f"{lu_cumrew[-1]:.0f}")
            m3.metric(
                "Oracle (best arm)",
                f"{oracle_rew[-1]:.0f}",
                delta=f"regret TS: {oracle_rew[-1] - ts_cumrew[-1]:.0f}",
                delta_color="inverse",
            )
        else:
            st.info("Нажмите **Запустить сравнение** для начала.")


# ===================================================================
# TAB 4 -- Кластеризация
# ===================================================================
with tab4:
    st.header("Кластеризация учеников (GMM + BIC)")
    st.markdown(
        "Генерируем синтетические mastery-вектора учеников, "
        "запускаем GMM с подбором k через BIC, визуализируем PCA-проекцию."
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
        if run_cluster:
            rng_cl = np.random.RandomState(42)

            # Генерируем данные из true_k гауссовых кластеров
            centers = rng_cl.rand(true_k, n_kc_dim).astype(np.float32)
            labels_true = rng_cl.randint(0, true_k, size=n_students)
            matrix = np.zeros((n_students, n_kc_dim), dtype=np.float32)
            for i in range(n_students):
                matrix[i] = centers[labels_true[i]] + rng_cl.randn(n_kc_dim).astype(np.float32) * 0.12

            matrix = np.clip(matrix, 0, 1)

            # Запускаем _select_n_clusters из проекта
            best_gmm, best_k = _select_n_clusters(matrix, max_k=max_k_search)
            labels_pred = best_gmm.predict(matrix)

            st.success(f"BIC выбрал k = **{best_k}** (истинное k = {true_k})")

            # --- BIC curve ---
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
                line=dict(color="#636EFA", width=2),
                marker=dict(size=8),
            ))
            fig_bic.add_vline(
                x=best_k, line_dash="dash", line_color="red",
                annotation_text=f"best k={best_k}",
            )
            fig_bic.update_layout(
                xaxis_title="k (число кластеров)", yaxis_title="BIC (ниже = лучше)",
                height=350,
            )
            st.plotly_chart(fig_bic, use_container_width=True)

            # --- PCA projection ---
            st.subheader("PCA-проекция (2D)")
            pca = PCA(n_components=2)
            coords = pca.fit_transform(matrix)

            fig_pca = px.scatter(
                x=coords[:, 0], y=coords[:, 1],
                color=[f"Кластер {l}" for l in labels_pred],
                title=f"Найдено {best_k} кластеров",
                labels={"x": "PC1", "y": "PC2", "color": "Кластер"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pca.update_layout(height=450)
            st.plotly_chart(fig_pca, use_container_width=True)

            # Cluster sizes
            st.subheader("Размеры кластеров")
            unique, counts = np.unique(labels_pred, return_counts=True)
            fig_sizes = px.bar(
                x=[f"Кластер {u}" for u in unique],
                y=counts,
                color=counts,
                color_continuous_scale="Viridis",
            )
            fig_sizes.update_layout(
                height=300, showlegend=False,
                xaxis_title="", yaxis_title="Число учеников",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_sizes, use_container_width=True)
        else:
            st.info("Нажмите **Запустить кластеризацию** для начала.")


# ===================================================================
# TAB 5 -- Архитектура системы
# ===================================================================
with tab5:
    st.header("Архитектура Learnity RecSys")
    st.markdown(
        "Микросервисная архитектура адаптивной рекомендательной системы "
        "для K-12 математического образования."
    )

    graph_dot = """
    digraph LearnityArchitecture {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11];
        edge [fontsize=9, fontname="Helvetica"];

        subgraph cluster_frontend {
            label="Frontend";
            style=dashed;
            color="#888888";
            fontcolor="#888888";
            Client [label="Web Client\\n(React)", fillcolor="#E8F4FD"];
        }

        subgraph cluster_gateway {
            label="API Gateway";
            style=dashed;
            color="#888888";
            fontcolor="#888888";
            Gateway [label="Gateway\\nFastAPI", fillcolor="#FFF3CD"];
        }

        subgraph cluster_services {
            label="ML Services";
            style=dashed;
            color="#888888";
            fontcolor="#888888";

            Profile [label="Profile Service\\nBKT / EMA mastery\\nConfidence tracking", fillcolor="#D4EDDA"];
            Retrieval [label="Retrieval Service\\nThompson Sampling\\nLinUCB / IRT / CAT", fillcolor="#D4EDDA"];
            Macro [label="Macro Service\\nPlan lifecycle\\nDiagnostics layer", fillcolor="#D4EDDA"];
            Graph [label="Graph Service\\nKC prerequisites\\nZPD computation", fillcolor="#CCE5FF"];
            Clustering [label="Clustering\\nGMM + BIC\\nStudent segments", fillcolor="#CCE5FF"];
        }

        subgraph cluster_data {
            label="Data Layer";
            style=dashed;
            color="#888888";
            fontcolor="#888888";
            PG [label="PostgreSQL\\nmastery, plans,\\nbandit models", fillcolor="#F5C6CB", shape=cylinder];
            Neo4j [label="Neo4j\\nKC graph", fillcolor="#F5C6CB", shape=cylinder];
            Kafka [label="Kafka\\nevents bus", fillcolor="#E2D5F1", shape=parallelogram];
        }

        Client -> Gateway [label="REST API"];
        Gateway -> Profile [label="mastery\\nupdate"];
        Gateway -> Retrieval [label="next task"];
        Gateway -> Macro [label="plan ops"];
        Retrieval -> Profile [label="get mastery"];
        Retrieval -> Graph [label="ZPD / prereqs"];
        Macro -> Profile [label="mastery read"];
        Macro -> Retrieval [label="recommend"];
        Macro -> Graph [label="prereqs"];
        Profile -> PG [label="read/write"];
        Retrieval -> PG [label="bandit models"];
        Macro -> PG [label="plans"];
        Graph -> Neo4j [label="KC graph"];
        Clustering -> PG [label="centroids"];
        Gateway -> Kafka [label="events"];
        Macro -> Kafka [label="transitions"];
    }
    """
    st.graphviz_chart(graph_dot, use_container_width=True)

    st.markdown("---")
    st.subheader("Ключевые компоненты")

    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        st.markdown("#### Profile Service")
        st.markdown(
            "- **BKT smooth_update** -- EMA-обновление mastery\n"
            "- **Confidence** -- надежность оценки (attempts, recency, stability)\n"
            "- **Decay** -- забывание: P *= 0.5^(days/half_life)\n"
            "- Поведенческие сигналы: guessing_rate, hint_dependence"
        )

    with col_a2:
        st.markdown("#### Retrieval Service")
        st.markdown(
            "- **Thompson Sampling** -- байесовский контекстный бандит\n"
            "- **LinUCB** -- детерминированный UCB-бандит (контрольная группа)\n"
            "- **IRT pre-filter** -- отсечение слишком простых/сложных заданий\n"
            "- **Diagnostic CAT** -- cold-start калибровка за 5-8 заданий\n"
            "- ZPD-фильтрация и subject rotation"
        )

    with col_a3:
        st.markdown("#### Macro Service")
        st.markdown(
            "- **Plan lifecycle** -- создание, продвижение, завершение планов\n"
            "- **MicroSummary** -- оценка прогресса на каждом шаге\n"
            "- **Diagnostic layer** -- prereq_gap, content_gap, regression, uncertain_estimate\n"
            "- **Transitions** -- логирование переходов для offline RL"
        )

    st.markdown("---")
    st.subheader("Diagnostic layer (причинный анализ)")
    st.markdown("Система определяет _причину_ затруднения ученика, а не просто реагирует на симптомы:")

    diag_examples = [
        {
            "label": "prereq_gap",
            "desc": "Слабый пререквизит тормозит освоение",
            "params": dict(
                mastery_current=0.3, velocity=-0.02, frustration_count=3,
                avg_score=0.3, tasks_spent=5, attempts_count=8,
                mastery_confidence=0.6, weakest_prereq_mastery=0.35, task_count_for_kc=10,
            ),
        },
        {
            "label": "uncertain_estimate",
            "desc": "Мало данных для надежной оценки",
            "params": dict(
                mastery_current=0.4, velocity=0.0, frustration_count=2,
                avg_score=0.4, tasks_spent=2, attempts_count=3,
                mastery_confidence=0.2, weakest_prereq_mastery=0.7, task_count_for_kc=10,
            ),
        },
        {
            "label": "regression",
            "desc": "Ученик забывает ранее освоенный материал",
            "params": dict(
                mastery_current=0.8, velocity=-0.05, frustration_count=3,
                avg_score=0.3, tasks_spent=4, attempts_count=15,
                mastery_confidence=0.7, weakest_prereq_mastery=0.8, task_count_for_kc=10,
            ),
        },
        {
            "label": "on_track",
            "desc": "Ученик нормально прогрессирует",
            "params": dict(
                mastery_current=0.5, velocity=0.03, frustration_count=0,
                avg_score=0.7, tasks_spent=3, attempts_count=8,
                mastery_confidence=0.6, weakest_prereq_mastery=0.8, task_count_for_kc=10,
            ),
        },
    ]

    for ex in diag_examples:
        d: Diagnosis = diagnose(**ex["params"])
        icon = {
            "prereq_gap": "!!",
            "content_gap": "?",
            "uncertain_estimate": "~",
            "regression": "<<",
            "on_track": "OK",
        }.get(d.reason, "?")
        color = {
            "prereq_gap": "red",
            "content_gap": "orange",
            "uncertain_estimate": "blue",
            "regression": "violet",
            "on_track": "green",
        }.get(d.reason, "gray")
        st.markdown(
            f":{color}[**[{icon}] {d.reason}**] (confidence={d.confidence:.2f}) -- {ex['desc']}  \n"
            f"> {d.detail}"
        )
