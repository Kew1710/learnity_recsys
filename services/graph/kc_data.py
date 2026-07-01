"""
Методический граф по математике 5–11 класс.

Чистые данные без внешних зависимостей — безопасно импортировать
из любого контекста (demo, tests, seed, etc.).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# KC nodes
# ---------------------------------------------------------------------------

KCS: list[dict] = [
    # ====================================================================
    # 5 класс — Арифметика
    # ====================================================================
    {
        "kc_id": "kc_natural_numbers",
        "name": "Натуральные числа. Ряд и координатный луч",
        "grade_introduced": 5, "difficulty_base": 0.05,
        "kc_type": "declarative", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_integer_arithmetic",
        "name": "Действия с натуральными числами (+−×÷)",
        "grade_introduced": 5, "difficulty_base": 0.10,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_variable_expression",
        "name": "Числовые и буквенные выражения. Формулы",
        "grade_introduced": 5, "difficulty_base": 0.15,
        "kc_type": "declarative", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_powers_natural",
        "name": "Степень числа с натуральным показателем",
        "grade_introduced": 5, "difficulty_base": 0.18,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_square_power",
        "name": "Возведение в квадрат",
        "grade_introduced": 5, "difficulty_base": 0.20,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_fractions_basic",
        "name": "Обыкновенные дроби: понятие, сравнение, смешанные числа",
        "grade_introduced": 5, "difficulty_base": 0.22,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_fractions_mul_div",
        "name": "Умножение и деление обыкновенных дробей",
        "grade_introduced": 5, "difficulty_base": 0.25,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_decimal_fractions",
        "name": "Десятичные дроби: запись, сравнение, действия",
        "grade_introduced": 5, "difficulty_base": 0.25,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_percents",
        "name": "Проценты. Решение задач",
        "grade_introduced": 5, "difficulty_base": 0.28,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_statistics_basic",
        "name": "Среднее арифметическое, медиана, размах",
        "grade_introduced": 5, "difficulty_base": 0.22,
        "kc_type": "procedural", "subject": "statistics",
    },

    # ====================================================================
    # 5 класс — Геометрия
    # ====================================================================
    {
        "kc_id": "kc_point_line_plane",
        "name": "Точка, прямая, луч, отрезок",
        "grade_introduced": 5, "difficulty_base": 0.08,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_angle_measure",
        "name": "Измерение углов. Транспортир",
        "grade_introduced": 5, "difficulty_base": 0.12,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_angle_types",
        "name": "Виды углов. Биссектриса угла",
        "grade_introduced": 5, "difficulty_base": 0.12,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_right_angle",
        "name": "Прямой угол. Перпендикулярность прямых",
        "grade_introduced": 5, "difficulty_base": 0.15,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_area_rectangle",
        "name": "Площадь прямоугольника и квадрата",
        "grade_introduced": 5, "difficulty_base": 0.20,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_triangle_basics",
        "name": "Треугольник: элементы, периметр",
        "grade_introduced": 5, "difficulty_base": 0.18,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_triangle_types",
        "name": "Виды треугольников",
        "grade_introduced": 5, "difficulty_base": 0.18,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_area_triangle",
        "name": "Площадь треугольника",
        "grade_introduced": 5, "difficulty_base": 0.22,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_circle_basics",
        "name": "Окружность и круг: элементы",
        "grade_introduced": 5, "difficulty_base": 0.18,
        "kc_type": "declarative", "subject": "geometry",
    },

    # ====================================================================
    # 6 класс — Арифметика
    # ====================================================================
    {
        "kc_id": "kc_divisibility",
        "name": "Делители и кратные. НОД, НОК. Простые числа",
        "grade_introduced": 6, "difficulty_base": 0.25,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_fractions_operations",
        "name": "Действия с дробями (разные знаменатели)",
        "grade_introduced": 6, "difficulty_base": 0.28,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_ratios_proportions",
        "name": "Отношения. Пропорции. Прямая и обратная зависимость",
        "grade_introduced": 6, "difficulty_base": 0.30,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_negative_numbers",
        "name": "Отрицательные числа. Модуль. Целые числа",
        "grade_introduced": 6, "difficulty_base": 0.28,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_rational_arithmetic",
        "name": "Действия с рациональными числами",
        "grade_introduced": 6, "difficulty_base": 0.32,
        "kc_type": "procedural", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_expression_simplify",
        "name": "Раскрытие скобок. Упрощение выражений",
        "grade_introduced": 6, "difficulty_base": 0.32,
        "kc_type": "procedural", "subject": "algebra",
    },

    # ====================================================================
    # 6 класс — Геометрия
    # ====================================================================
    {
        "kc_id": "kc_circle_length_area",
        "name": "Длина окружности и площадь круга",
        "grade_introduced": 6, "difficulty_base": 0.30,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_coordinate_plane",
        "name": "Координатная плоскость. Графики",
        "grade_introduced": 6, "difficulty_base": 0.30,
        "kc_type": "procedural", "subject": "geometry",
    },

    # ====================================================================
    # 7 класс — Алгебра
    # ====================================================================
    {
        "kc_id": "kc_linear_eq_1var",
        "name": "Линейное уравнение с одной переменной",
        "grade_introduced": 7, "difficulty_base": 0.38,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_function_concept",
        "name": "Понятие функции. Область определения",
        "grade_introduced": 7, "difficulty_base": 0.38,
        "kc_type": "declarative", "subject": "algebra",
    },
    {
        "kc_id": "kc_linear_function",
        "name": "Линейная функция y=kx+b",
        "grade_introduced": 7, "difficulty_base": 0.42,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_linear_function_graph",
        "name": "График линейной функции",
        "grade_introduced": 7, "difficulty_base": 0.45,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_linear_eq_system",
        "name": "Система линейных уравнений",
        "grade_introduced": 7, "difficulty_base": 0.48,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_system_substitution",
        "name": "Метод подстановки",
        "grade_introduced": 7, "difficulty_base": 0.50,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_system_addition",
        "name": "Метод алгебраического сложения",
        "grade_introduced": 7, "difficulty_base": 0.50,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_powers_properties",
        "name": "Свойства степени с натуральным показателем",
        "grade_introduced": 7, "difficulty_base": 0.40,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_monomial",
        "name": "Одночлен: стандартный вид, действия",
        "grade_introduced": 7, "difficulty_base": 0.40,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_polynomial",
        "name": "Многочлен: понятие, степень",
        "grade_introduced": 7, "difficulty_base": 0.42,
        "kc_type": "declarative", "subject": "algebra",
    },
    {
        "kc_id": "kc_polynomial_operations",
        "name": "Действия с многочленами",
        "grade_introduced": 7, "difficulty_base": 0.45,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_factoring_common",
        "name": "Вынесение общего множителя за скобки",
        "grade_introduced": 7, "difficulty_base": 0.48,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_factoring_grouping",
        "name": "Метод группировки",
        "grade_introduced": 7, "difficulty_base": 0.50,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_factoring_formulas",
        "name": "Формулы сокращённого умножения",
        "grade_introduced": 7, "difficulty_base": 0.52,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_algebraic_fraction",
        "name": "Алгебраическая дробь. Сокращение",
        "grade_introduced": 7, "difficulty_base": 0.52,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_algebraic_fraction_ops",
        "name": "Действия с алгебраическими дробями",
        "grade_introduced": 7, "difficulty_base": 0.55,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_word_problems_eq",
        "name": "Текстовые задачи через уравнения",
        "grade_introduced": 7, "difficulty_base": 0.58,
        "kc_type": "applied", "subject": "algebra",
    },
    {
        "kc_id": "kc_combinatorics_basic",
        "name": "Элементы комбинаторики",
        "grade_introduced": 7, "difficulty_base": 0.48,
        "kc_type": "procedural", "subject": "statistics",
    },

    # ====================================================================
    # 7 класс — Геометрия
    # ====================================================================
    {
        "kc_id": "kc_parallel_lines",
        "name": "Параллельные прямые. Признаки и аксиома",
        "grade_introduced": 7, "difficulty_base": 0.38,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_parallel_angles",
        "name": "Углы при параллельных прямых",
        "grade_introduced": 7, "difficulty_base": 0.42,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_triangle_angle_sum",
        "name": "Сумма углов треугольника",
        "grade_introduced": 7, "difficulty_base": 0.38,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_triangle_congruence",
        "name": "Признаки равенства треугольников",
        "grade_introduced": 7, "difficulty_base": 0.45,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_triangle_elements",
        "name": "Медиана, биссектриса, высота треугольника",
        "grade_introduced": 7, "difficulty_base": 0.42,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_isosceles_triangle",
        "name": "Равнобедренный треугольник",
        "grade_introduced": 7, "difficulty_base": 0.45,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_right_triangle_parts",
        "name": "Катет и гипотенуза прямоугольного треугольника",
        "grade_introduced": 7, "difficulty_base": 0.38,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_triangle_sides_angles",
        "name": "Соотношения между сторонами и углами треугольника",
        "grade_introduced": 7, "difficulty_base": 0.48,
        "kc_type": "procedural", "subject": "geometry",
    },

    # ====================================================================
    # 8 класс — Алгебра
    # ====================================================================
    {
        "kc_id": "kc_powers_integer_exp",
        "name": "Степень с целым (отрицательным) показателем",
        "grade_introduced": 8, "difficulty_base": 0.48,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_inverse_proportion",
        "name": "Обратная пропорциональность y=k/x и её график",
        "grade_introduced": 8, "difficulty_base": 0.52,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_sqrt_concept",
        "name": "Арифметический квадратный корень. Свойства",
        "grade_introduced": 8, "difficulty_base": 0.50,
        "kc_type": "declarative", "subject": "algebra",
    },
    {
        "kc_id": "kc_sqrt_compute",
        "name": "Вычисление и оценка квадратных корней. Иррациональные числа",
        "grade_introduced": 8, "difficulty_base": 0.55,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_sqrt_simplify",
        "name": "Преобразования выражений с квадратными корнями",
        "grade_introduced": 8, "difficulty_base": 0.58,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_sqrt_function",
        "name": "Функция y=√x и её свойства",
        "grade_introduced": 8, "difficulty_base": 0.58,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_quadratic_function",
        "name": "Квадратичная функция y=ax²+bx+c",
        "grade_introduced": 8, "difficulty_base": 0.60,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_quadratic_func_graph",
        "name": "График квадратичной функции. Парабола",
        "grade_introduced": 8, "difficulty_base": 0.62,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_quadratic_eq",
        "name": "Квадратное уравнение",
        "grade_introduced": 8, "difficulty_base": 0.55,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_quadratic_discriminant",
        "name": "Дискриминант квадратного уравнения",
        "grade_introduced": 8, "difficulty_base": 0.55,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_quadratic_vieta",
        "name": "Теорема Виета",
        "grade_introduced": 8, "difficulty_base": 0.58,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_quadratic_factoring",
        "name": "Разложение квадратного трёхчлена на множители",
        "grade_introduced": 8, "difficulty_base": 0.60,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_biquadratic_eq",
        "name": "Биквадратные и дробно-рациональные уравнения",
        "grade_introduced": 8, "difficulty_base": 0.65,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_rational_eq",
        "name": "Дробно-рациональное уравнение",
        "grade_introduced": 8, "difficulty_base": 0.65,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_irrational_eq",
        "name": "Иррациональные уравнения",
        "grade_introduced": 8, "difficulty_base": 0.65,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_inequality_props",
        "name": "Свойства числовых неравенств",
        "grade_introduced": 8, "difficulty_base": 0.45,
        "kc_type": "declarative", "subject": "algebra",
    },
    {
        "kc_id": "kc_linear_inequality",
        "name": "Линейное неравенство. Решение на числовой прямой",
        "grade_introduced": 8, "difficulty_base": 0.50,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_absolute_value_eq",
        "name": "Модуль числа. Уравнения и неравенства с модулем",
        "grade_introduced": 8, "difficulty_base": 0.58,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_quadratic_inequality",
        "name": "Квадратное неравенство",
        "grade_introduced": 8, "difficulty_base": 0.62,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_interval_method",
        "name": "Метод интервалов",
        "grade_introduced": 8, "difficulty_base": 0.65,
        "kc_type": "procedural", "subject": "algebra",
    },

    # ====================================================================
    # 8 класс — Геометрия
    # ====================================================================
    {
        "kc_id": "kc_quadrilaterals",
        "name": "Параллелограмм, трапеция, прямоугольник, ромб, квадрат",
        "grade_introduced": 8, "difficulty_base": 0.48,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_area_parallelogram",
        "name": "Площадь параллелограмма и трапеции",
        "grade_introduced": 8, "difficulty_base": 0.52,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_pythagorean_know",
        "name": "Теорема Пифагора: a²+b²=c²",
        "grade_introduced": 8, "difficulty_base": 0.55,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_pythagorean_find_hyp",
        "name": "Нахождение гипотенузы",
        "grade_introduced": 8, "difficulty_base": 0.58,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_pythagorean_find_leg",
        "name": "Нахождение катета",
        "grade_introduced": 8, "difficulty_base": 0.60,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_heron_formula",
        "name": "Формула Герона",
        "grade_introduced": 8, "difficulty_base": 0.62,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_triangle_similarity",
        "name": "Признаки подобия треугольников",
        "grade_introduced": 8, "difficulty_base": 0.58,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_trig_sin_cos_tan",
        "name": "Синус, косинус, тангенс угла прямоугольного треугольника",
        "grade_introduced": 8, "difficulty_base": 0.60,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_trig_compute",
        "name": "Вычисления с тригонометрией",
        "grade_introduced": 8, "difficulty_base": 0.65,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_circle_tangent",
        "name": "Касательная к окружности. Вписанный угол",
        "grade_introduced": 8, "difficulty_base": 0.58,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_circle_properties",
        "name": "Центральные и вписанные углы. Свойства хорд",
        "grade_introduced": 8, "difficulty_base": 0.62,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_inscribed_circumscribed",
        "name": "Вписанная и описанная окружности треугольника",
        "grade_introduced": 8, "difficulty_base": 0.65,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_pythagorean_word",
        "name": "Задачи на теорему Пифагора",
        "grade_introduced": 8, "difficulty_base": 0.68,
        "kc_type": "applied", "subject": "geometry",
    },

    # ====================================================================
    # 8 класс — Статистика
    # ====================================================================
    {
        "kc_id": "kc_probability_basic",
        "name": "Вероятность: понятие и классическая формула",
        "grade_introduced": 8, "difficulty_base": 0.48,
        "kc_type": "declarative", "subject": "statistics",
    },
    {
        "kc_id": "kc_probability_compute",
        "name": "Вычисление вероятностей",
        "grade_introduced": 8, "difficulty_base": 0.55,
        "kc_type": "procedural", "subject": "statistics",
    },

    # ====================================================================
    # 9 класс — Алгебра
    # ====================================================================
    {
        "kc_id": "kc_function_properties",
        "name": "Возрастание/убывание, чётность/нечётность функции",
        "grade_introduced": 9, "difficulty_base": 0.58,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_sequence_concept",
        "name": "Числовая последовательность",
        "grade_introduced": 9, "difficulty_base": 0.55,
        "kc_type": "declarative", "subject": "algebra",
    },
    {
        "kc_id": "kc_arithmetic_progression",
        "name": "Арифметическая прогрессия",
        "grade_introduced": 9, "difficulty_base": 0.60,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_arithmetic_progression_sum",
        "name": "Сумма n первых членов арифметической прогрессии",
        "grade_introduced": 9, "difficulty_base": 0.62,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_geometric_progression",
        "name": "Геометрическая прогрессия",
        "grade_introduced": 9, "difficulty_base": 0.65,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_geometric_progression_sum",
        "name": "Сумма n первых членов геометрической прогрессии",
        "grade_introduced": 9, "difficulty_base": 0.68,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_inequality_system",
        "name": "Система линейных неравенств",
        "grade_introduced": 9, "difficulty_base": 0.62,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_coordinate_geometry_9",
        "name": "Уравнение окружности и прямой на плоскости",
        "grade_introduced": 9, "difficulty_base": 0.65,
        "kc_type": "procedural", "subject": "algebra",
    },

    # ====================================================================
    # 9 класс — Статистика
    # ====================================================================
    {
        "kc_id": "kc_probability_rules",
        "name": "Сложение и умножение вероятностей",
        "grade_introduced": 9, "difficulty_base": 0.62,
        "kc_type": "procedural", "subject": "statistics",
    },
    {
        "kc_id": "kc_combinatorics_probability",
        "name": "Решение вероятностных задач с помощью комбинаторики",
        "grade_introduced": 9, "difficulty_base": 0.68,
        "kc_type": "applied", "subject": "statistics",
    },

    # ====================================================================
    # 9 класс — Геометрия
    # ====================================================================
    {
        "kc_id": "kc_vectors",
        "name": "Векторы. Сложение, вычитание, умножение на число",
        "grade_introduced": 9, "difficulty_base": 0.58,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_coordinate_method",
        "name": "Метод координат в геометрии",
        "grade_introduced": 9, "difficulty_base": 0.62,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_trig_basic_identity",
        "name": "Основное тригонометрическое тождество. Формулы",
        "grade_introduced": 9, "difficulty_base": 0.62,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_trig_theorems",
        "name": "Теорема синусов и теорема косинусов",
        "grade_introduced": 9, "difficulty_base": 0.68,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_regular_polygon",
        "name": "Правильные многоугольники. Вписанная и описанная окружности",
        "grade_introduced": 9, "difficulty_base": 0.68,
        "kc_type": "procedural", "subject": "geometry",
    },

    # ====================================================================
    # 10 класс — Алгебра
    # ====================================================================
    {
        "kc_id": "kc_real_numbers",
        "name": "Множество действительных чисел. Иррациональные числа",
        "grade_introduced": 10, "difficulty_base": 0.55,
        "kc_type": "declarative", "subject": "arithmetic",
    },
    {
        "kc_id": "kc_function_transformations",
        "name": "Построение графиков с помощью геометрических преобразований",
        "grade_introduced": 10, "difficulty_base": 0.60,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_inverse_function",
        "name": "Обратная функция",
        "grade_introduced": 10, "difficulty_base": 0.62,
        "kc_type": "declarative", "subject": "algebra",
    },
    {
        "kc_id": "kc_power_function_nat",
        "name": "Степенная функция y=x^n (натуральный показатель)",
        "grade_introduced": 10, "difficulty_base": 0.58,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_power_function_int",
        "name": "Степенная функция y=x^n (целый показатель)",
        "grade_introduced": 10, "difficulty_base": 0.60,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_nth_root",
        "name": "Корень n-й степени. Функция корня n-й степени",
        "grade_introduced": 10, "difficulty_base": 0.62,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_rational_power",
        "name": "Степень с рациональным показателем. Свойства",
        "grade_introduced": 10, "difficulty_base": 0.65,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_irrational_ineq",
        "name": "Иррациональные неравенства",
        "grade_introduced": 10, "difficulty_base": 0.68,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_exponential_eq",
        "name": "Показательные уравнения",
        "grade_introduced": 10, "difficulty_base": 0.65,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_exponential_ineq",
        "name": "Показательные неравенства",
        "grade_introduced": 10, "difficulty_base": 0.68,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_trig_radian",
        "name": "Радианная мера угла. Числовая окружность",
        "grade_introduced": 10, "difficulty_base": 0.62,
        "kc_type": "declarative", "subject": "algebra",
    },
    {
        "kc_id": "kc_trig_functions_props",
        "name": "Свойства и графики тригонометрических функций",
        "grade_introduced": 10, "difficulty_base": 0.68,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_trig_identities",
        "name": "Тригонометрические тождества",
        "grade_introduced": 10, "difficulty_base": 0.68,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_trig_addition_formulas",
        "name": "Формулы сложения, двойного и половинного угла",
        "grade_introduced": 10, "difficulty_base": 0.72,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_trig_eq_ineq",
        "name": "Тригонометрические уравнения и неравенства",
        "grade_introduced": 10, "difficulty_base": 0.75,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_limit_concept",
        "name": "Предел числовой последовательности и функции",
        "grade_introduced": 10, "difficulty_base": 0.72,
        "kc_type": "declarative", "subject": "algebra",
    },
    {
        "kc_id": "kc_derivative_concept",
        "name": "Понятие производной. Вычисление производных",
        "grade_introduced": 10, "difficulty_base": 0.72,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_derivative_rules",
        "name": "Дифференцирование сложных и обратных функций",
        "grade_introduced": 10, "difficulty_base": 0.75,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_tangent_equation",
        "name": "Уравнение касательной к графику функции",
        "grade_introduced": 10, "difficulty_base": 0.75,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_derivative_analysis",
        "name": "Исследование функции с помощью производной",
        "grade_introduced": 10, "difficulty_base": 0.78,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_extremum_values",
        "name": "Нахождение наибольших и наименьших значений функции",
        "grade_introduced": 10, "difficulty_base": 0.80,
        "kc_type": "applied", "subject": "algebra",
    },

    # ====================================================================
    # 10 класс — Геометрия
    # ====================================================================
    {
        "kc_id": "kc_polyhedron",
        "name": "Многогранники: призма и пирамида",
        "grade_introduced": 10, "difficulty_base": 0.60,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_regular_polyhedron",
        "name": "Правильные многогранники",
        "grade_introduced": 10, "difficulty_base": 0.65,
        "kc_type": "declarative", "subject": "geometry",
    },
    {
        "kc_id": "kc_space_lines_planes",
        "name": "Параллельность прямых и плоскостей в пространстве",
        "grade_introduced": 10, "difficulty_base": 0.65,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_perpendicularity_3d",
        "name": "Перпендикулярность прямых и плоскостей",
        "grade_introduced": 10, "difficulty_base": 0.68,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_vectors_3d",
        "name": "Векторы и координаты в пространстве. Скалярное произведение",
        "grade_introduced": 10, "difficulty_base": 0.68,
        "kc_type": "procedural", "subject": "geometry",
    },

    # ====================================================================
    # 11 класс — Алгебра
    # ====================================================================
    {
        "kc_id": "kc_logarithm_concept",
        "name": "Логарифмы и их свойства",
        "grade_introduced": 11, "difficulty_base": 0.70,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_logarithm_function",
        "name": "Логарифмическая функция: свойства и график",
        "grade_introduced": 11, "difficulty_base": 0.72,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_logarithmic_eq",
        "name": "Логарифмические уравнения",
        "grade_introduced": 11, "difficulty_base": 0.75,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_logarithmic_ineq",
        "name": "Логарифмические неравенства",
        "grade_introduced": 11, "difficulty_base": 0.78,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_higher_degree_eq",
        "name": "Уравнения высших степеней",
        "grade_introduced": 11, "difficulty_base": 0.75,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_antiderivative",
        "name": "Первообразная и неопределённый интеграл",
        "grade_introduced": 11, "difficulty_base": 0.78,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_definite_integral",
        "name": "Определённый интеграл. Вычисление площадей",
        "grade_introduced": 11, "difficulty_base": 0.82,
        "kc_type": "procedural", "subject": "algebra",
    },
    {
        "kc_id": "kc_permutations_combinations",
        "name": "Перестановки, размещения, сочетания",
        "grade_introduced": 11, "difficulty_base": 0.70,
        "kc_type": "procedural", "subject": "statistics",
    },
    {
        "kc_id": "kc_binomial_theorem",
        "name": "Бином Ньютона",
        "grade_introduced": 11, "difficulty_base": 0.75,
        "kc_type": "procedural", "subject": "statistics",
    },
    {
        "kc_id": "kc_parametric_problems",
        "name": "Задачи с параметрами",
        "grade_introduced": 11, "difficulty_base": 0.88,
        "kc_type": "applied", "subject": "algebra",
    },

    # ====================================================================
    # 11 класс — Геометрия (стереометрия)
    # ====================================================================
    {
        "kc_id": "kc_cylinder_cone",
        "name": "Цилиндр и конус: элементы, площадь поверхности",
        "grade_introduced": 11, "difficulty_base": 0.68,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_sphere",
        "name": "Шар и сфера. Площадь поверхности шара",
        "grade_introduced": 11, "difficulty_base": 0.70,
        "kc_type": "procedural", "subject": "geometry",
    },
    {
        "kc_id": "kc_volumes_solids",
        "name": "Объёмы многогранников и тел вращения",
        "grade_introduced": 11, "difficulty_base": 0.75,
        "kc_type": "procedural", "subject": "geometry",
    },
]

# ---------------------------------------------------------------------------
# Edges: (from_kc, to_kc, strength)
# ---------------------------------------------------------------------------

EDGES: list[tuple[str, str, float]] = [
    # ================================================================
    # Арифметика 5 класс
    # ================================================================
    ("kc_natural_numbers",        "kc_integer_arithmetic",      0.9),
    ("kc_natural_numbers",        "kc_fractions_basic",         0.8),
    ("kc_integer_arithmetic",     "kc_variable_expression",     0.9),
    ("kc_integer_arithmetic",     "kc_powers_natural",          0.8),
    ("kc_integer_arithmetic",     "kc_decimal_fractions",       0.8),
    ("kc_integer_arithmetic",     "kc_statistics_basic",        0.8),
    ("kc_fractions_basic",        "kc_fractions_mul_div",       0.9),
    ("kc_fractions_basic",        "kc_decimal_fractions",       0.7),
    ("kc_fractions_mul_div",      "kc_percents",                0.8),
    ("kc_decimal_fractions",      "kc_percents",                0.8),
    ("kc_powers_natural",         "kc_square_power",            0.95),

    # ================================================================
    # Геометрия 5 класс
    # ================================================================
    ("kc_point_line_plane",       "kc_angle_measure",           0.9),
    ("kc_angle_measure",          "kc_angle_types",             0.9),
    ("kc_angle_types",            "kc_right_angle",             0.9),
    ("kc_angle_types",            "kc_triangle_basics",         0.8),
    ("kc_angle_types",            "kc_triangle_types",          0.7),
    ("kc_right_angle",            "kc_triangle_types",          0.7),
    ("kc_triangle_basics",        "kc_triangle_types",          0.9),
    ("kc_triangle_basics",        "kc_area_triangle",           0.8),
    ("kc_point_line_plane",       "kc_area_rectangle",          0.7),
    ("kc_integer_arithmetic",     "kc_area_rectangle",          0.6),
    ("kc_area_rectangle",         "kc_area_triangle",           0.8),
    ("kc_point_line_plane",       "kc_circle_basics",           0.8),

    # ================================================================
    # Арифметика 6 класс
    # ================================================================
    ("kc_integer_arithmetic",     "kc_divisibility",            0.8),
    ("kc_fractions_basic",        "kc_fractions_operations",    0.9),
    ("kc_integer_arithmetic",     "kc_fractions_operations",    0.8),
    ("kc_divisibility",           "kc_fractions_operations",    0.4),
    ("kc_fractions_operations",   "kc_ratios_proportions",      0.8),
    ("kc_fractions_mul_div",      "kc_ratios_proportions",      0.7),
    ("kc_integer_arithmetic",     "kc_negative_numbers",        0.9),
    ("kc_negative_numbers",       "kc_rational_arithmetic",     0.9),
    ("kc_fractions_operations",   "kc_rational_arithmetic",     0.8),
    ("kc_variable_expression",    "kc_expression_simplify",     0.9),
    ("kc_rational_arithmetic",    "kc_expression_simplify",     0.7),

    # ================================================================
    # Геометрия 6 класс
    # ================================================================
    ("kc_circle_basics",          "kc_circle_length_area",      0.9),
    ("kc_decimal_fractions",      "kc_circle_length_area",      0.5),
    ("kc_negative_numbers",       "kc_coordinate_plane",        0.9),
    ("kc_integer_arithmetic",     "kc_coordinate_plane",        0.6),

    # ================================================================
    # Алгебра 7 класс
    # ================================================================
    ("kc_variable_expression",    "kc_linear_eq_1var",          0.8),
    ("kc_rational_arithmetic",    "kc_linear_eq_1var",          0.8),
    ("kc_expression_simplify",    "kc_linear_eq_1var",          0.7),
    ("kc_linear_eq_1var",         "kc_linear_eq_system",        0.9),
    ("kc_linear_eq_system",       "kc_system_substitution",     0.9),
    ("kc_linear_eq_system",       "kc_system_addition",         0.9),
    ("kc_variable_expression",    "kc_function_concept",        0.8),
    ("kc_coordinate_plane",       "kc_function_concept",        0.7),
    ("kc_function_concept",       "kc_linear_function",         0.9),
    ("kc_linear_eq_1var",         "kc_linear_function",         0.7),
    ("kc_linear_function",        "kc_linear_function_graph",   0.95),
    ("kc_coordinate_plane",       "kc_linear_function_graph",   0.9),
    ("kc_powers_natural",         "kc_powers_properties",       0.9),
    ("kc_powers_properties",      "kc_monomial",                0.9),
    ("kc_variable_expression",    "kc_monomial",                0.7),
    ("kc_monomial",               "kc_polynomial",              0.9),
    ("kc_polynomial",             "kc_polynomial_operations",   0.95),
    ("kc_polynomial_operations",  "kc_factoring_common",        0.9),
    ("kc_polynomial_operations",  "kc_factoring_grouping",      0.8),
    ("kc_factoring_grouping",     "kc_factoring_formulas",      0.7),
    ("kc_polynomial_operations",  "kc_factoring_formulas",      0.8),
    ("kc_square_power",           "kc_factoring_formulas",      0.8),
    ("kc_factoring_common",       "kc_algebraic_fraction",      0.8),
    ("kc_factoring_formulas",     "kc_algebraic_fraction",      0.9),
    ("kc_algebraic_fraction",     "kc_algebraic_fraction_ops",  0.95),
    ("kc_fractions_operations",   "kc_algebraic_fraction_ops",  0.5),
    ("kc_linear_eq_1var",         "kc_word_problems_eq",        0.9),
    ("kc_percents",               "kc_word_problems_eq",        0.5),
    ("kc_rational_arithmetic",    "kc_combinatorics_basic",     0.7),

    # ================================================================
    # Геометрия 7 класс
    # ================================================================
    ("kc_angle_types",            "kc_parallel_lines",          0.8),
    ("kc_parallel_lines",         "kc_parallel_angles",         0.95),
    ("kc_parallel_angles",        "kc_triangle_angle_sum",      0.8),
    ("kc_triangle_basics",        "kc_triangle_angle_sum",      0.9),
    ("kc_triangle_basics",        "kc_triangle_congruence",     0.9),
    ("kc_angle_types",            "kc_triangle_congruence",     0.7),
    ("kc_triangle_congruence",    "kc_isosceles_triangle",      0.9),
    ("kc_triangle_basics",        "kc_triangle_elements",       0.8),
    ("kc_triangle_congruence",    "kc_triangle_elements",       0.7),
    ("kc_triangle_angle_sum",     "kc_triangle_sides_angles",   0.8),
    ("kc_parallel_angles",        "kc_triangle_sides_angles",   0.6),
    ("kc_right_angle",            "kc_right_triangle_parts",    0.9),
    ("kc_triangle_types",         "kc_right_triangle_parts",    0.8),

    # ================================================================
    # Алгебра 8 класс
    # ================================================================
    ("kc_powers_properties",      "kc_powers_integer_exp",      0.8),
    ("kc_negative_numbers",       "kc_powers_integer_exp",      0.7),
    ("kc_function_concept",       "kc_inverse_proportion",      0.7),
    ("kc_rational_arithmetic",    "kc_inverse_proportion",      0.6),
    ("kc_square_power",           "kc_sqrt_concept",            0.95),
    ("kc_sqrt_concept",           "kc_sqrt_compute",            0.9),
    ("kc_sqrt_compute",           "kc_sqrt_simplify",           0.8),
    ("kc_factoring_common",       "kc_sqrt_simplify",           0.6),
    ("kc_sqrt_simplify",          "kc_sqrt_function",           0.8),
    ("kc_function_concept",       "kc_sqrt_function",           0.7),
    ("kc_polynomial_operations",  "kc_quadratic_eq",            0.8),
    ("kc_sqrt_compute",           "kc_quadratic_eq",            0.8),
    ("kc_quadratic_eq",           "kc_quadratic_discriminant",  0.95),
    ("kc_quadratic_eq",           "kc_quadratic_vieta",         0.9),
    ("kc_quadratic_vieta",        "kc_quadratic_factoring",     0.9),
    ("kc_factoring_formulas",     "kc_quadratic_factoring",     0.7),
    ("kc_quadratic_eq",           "kc_biquadratic_eq",          0.8),
    ("kc_algebraic_fraction_ops", "kc_biquadratic_eq",          0.6),
    ("kc_algebraic_fraction_ops", "kc_rational_eq",             0.9),
    ("kc_linear_eq_1var",         "kc_rational_eq",             0.7),
    ("kc_sqrt_simplify",          "kc_irrational_eq",           0.9),
    ("kc_quadratic_eq",           "kc_irrational_eq",           0.7),
    ("kc_quadratic_eq",           "kc_quadratic_function",      0.8),
    ("kc_linear_function",        "kc_quadratic_function",      0.6),
    ("kc_quadratic_function",     "kc_quadratic_func_graph",    0.95),
    ("kc_rational_arithmetic",    "kc_inequality_props",        0.8),
    ("kc_inequality_props",       "kc_linear_inequality",       0.8),
    ("kc_linear_eq_1var",         "kc_linear_inequality",       0.7),
    ("kc_linear_inequality",      "kc_inequality_system",       0.9),
    ("kc_linear_eq_1var",         "kc_absolute_value_eq",       0.8),
    ("kc_linear_inequality",      "kc_absolute_value_eq",       0.7),
    ("kc_quadratic_factoring",    "kc_interval_method",         0.8),
    ("kc_interval_method",        "kc_quadratic_inequality",    0.9),
    ("kc_quadratic_eq",           "kc_quadratic_inequality",    0.7),

    # ================================================================
    # Геометрия 8 класс
    # ================================================================
    ("kc_parallel_lines",         "kc_quadrilaterals",          0.8),
    ("kc_triangle_basics",        "kc_quadrilaterals",          0.6),
    ("kc_quadrilaterals",         "kc_area_parallelogram",      0.9),
    ("kc_area_rectangle",         "kc_area_parallelogram",      0.8),
    ("kc_area_triangle",          "kc_area_parallelogram",      0.7),
    ("kc_right_triangle_parts",   "kc_pythagorean_know",        0.9),
    ("kc_square_power",           "kc_pythagorean_know",        0.8),
    ("kc_sqrt_concept",           "kc_pythagorean_know",        0.7),
    ("kc_pythagorean_know",       "kc_pythagorean_find_hyp",    0.95),
    ("kc_sqrt_compute",           "kc_pythagorean_find_hyp",    0.8),
    ("kc_pythagorean_find_hyp",   "kc_pythagorean_find_leg",    0.9),
    ("kc_pythagorean_find_hyp",   "kc_pythagorean_word",        0.8),
    ("kc_pythagorean_find_leg",   "kc_pythagorean_word",        0.9),
    ("kc_area_triangle",          "kc_heron_formula",           0.8),
    ("kc_sqrt_compute",           "kc_heron_formula",           0.7),
    ("kc_triangle_congruence",    "kc_triangle_similarity",     0.8),
    ("kc_fractions_basic",        "kc_triangle_similarity",     0.5),
    ("kc_right_triangle_parts",   "kc_trig_sin_cos_tan",        0.9),
    ("kc_ratios_proportions",     "kc_trig_sin_cos_tan",        0.5),
    ("kc_trig_sin_cos_tan",       "kc_trig_compute",            0.9),
    ("kc_pythagorean_find_hyp",   "kc_trig_compute",            0.7),
    ("kc_circle_basics",          "kc_circle_tangent",          0.8),
    ("kc_right_angle",            "kc_circle_tangent",          0.7),
    ("kc_circle_tangent",         "kc_circle_properties",       0.8),
    ("kc_triangle_congruence",    "kc_circle_properties",       0.7),
    ("kc_circle_properties",      "kc_inscribed_circumscribed", 0.9),
    ("kc_triangle_congruence",    "kc_inscribed_circumscribed", 0.6),

    # ================================================================
    # Статистика 8 класс
    # ================================================================
    ("kc_combinatorics_basic",    "kc_probability_basic",       0.7),
    ("kc_fractions_basic",        "kc_probability_basic",       0.8),
    ("kc_probability_basic",      "kc_probability_compute",     0.95),
    ("kc_rational_arithmetic",    "kc_probability_compute",     0.6),

    # ================================================================
    # Алгебра 9 класс
    # ================================================================
    ("kc_function_concept",       "kc_function_properties",     0.8),
    ("kc_quadratic_function",     "kc_function_properties",     0.7),
    ("kc_function_concept",       "kc_sequence_concept",        0.6),
    ("kc_sequence_concept",       "kc_arithmetic_progression",  0.9),
    ("kc_linear_function",        "kc_arithmetic_progression",  0.5),
    ("kc_arithmetic_progression", "kc_arithmetic_progression_sum", 0.95),
    ("kc_sequence_concept",       "kc_geometric_progression",   0.9),
    ("kc_powers_natural",         "kc_geometric_progression",   0.6),
    ("kc_geometric_progression",  "kc_geometric_progression_sum", 0.95),
    ("kc_linear_inequality",      "kc_inequality_system",       0.9),
    ("kc_coordinate_plane",       "kc_coordinate_geometry_9",   0.8),
    ("kc_quadratic_function",     "kc_coordinate_geometry_9",   0.6),

    # ================================================================
    # Статистика 9 класс
    # ================================================================
    ("kc_probability_compute",    "kc_probability_rules",       0.9),
    ("kc_combinatorics_basic",    "kc_combinatorics_probability", 0.9),
    ("kc_probability_rules",      "kc_combinatorics_probability", 0.7),

    # ================================================================
    # Геометрия 9 класс
    # ================================================================
    ("kc_coordinate_plane",       "kc_vectors",                 0.8),
    ("kc_rational_arithmetic",    "kc_vectors",                 0.6),
    ("kc_vectors",                "kc_coordinate_method",       0.9),
    ("kc_coordinate_plane",       "kc_coordinate_method",       0.8),
    ("kc_trig_sin_cos_tan",       "kc_trig_basic_identity",     0.9),
    ("kc_pythagorean_find_hyp",   "kc_trig_basic_identity",     0.7),
    ("kc_trig_compute",           "kc_trig_theorems",           0.9),
    ("kc_trig_basic_identity",    "kc_trig_theorems",           0.8),
    ("kc_triangle_similarity",    "kc_trig_theorems",           0.7),
    ("kc_circle_properties",      "kc_regular_polygon",         0.8),
    ("kc_trig_compute",           "kc_regular_polygon",         0.7),

    # ================================================================
    # Алгебра 10 класс
    # ================================================================
    ("kc_sqrt_simplify",          "kc_real_numbers",            0.7),
    ("kc_irrational_eq",          "kc_real_numbers",            0.7),
    ("kc_function_concept",       "kc_function_transformations",0.8),
    ("kc_quadratic_func_graph",   "kc_function_transformations",0.7),
    ("kc_function_concept",       "kc_inverse_function",        0.8),
    ("kc_linear_function",        "kc_inverse_function",        0.6),
    ("kc_powers_natural",         "kc_power_function_nat",      0.8),
    ("kc_function_concept",       "kc_power_function_nat",      0.7),
    ("kc_power_function_nat",     "kc_power_function_int",      0.9),
    ("kc_powers_integer_exp",     "kc_power_function_int",      0.8),
    ("kc_sqrt_compute",           "kc_nth_root",                0.8),
    ("kc_power_function_nat",     "kc_nth_root",                0.7),
    ("kc_nth_root",               "kc_rational_power",          0.9),
    ("kc_powers_integer_exp",     "kc_rational_power",          0.8),
    ("kc_irrational_eq",          "kc_irrational_ineq",         0.8),
    ("kc_interval_method",        "kc_irrational_ineq",         0.7),
    ("kc_power_function_nat",     "kc_exponential_eq",          0.7),
    ("kc_rational_arithmetic",    "kc_exponential_eq",          0.6),
    ("kc_exponential_eq",         "kc_exponential_ineq",        0.9),
    ("kc_linear_inequality",      "kc_exponential_ineq",        0.6),
    ("kc_trig_sin_cos_tan",       "kc_trig_radian",             0.8),
    ("kc_function_concept",       "kc_trig_radian",             0.7),
    ("kc_trig_radian",            "kc_trig_functions_props",    0.9),
    ("kc_function_properties",    "kc_trig_functions_props",    0.7),
    ("kc_trig_basic_identity",    "kc_trig_identities",         0.9),
    ("kc_trig_functions_props",   "kc_trig_identities",         0.8),
    ("kc_trig_identities",        "kc_trig_addition_formulas",  0.9),
    ("kc_trig_addition_formulas", "kc_trig_eq_ineq",            0.9),
    ("kc_trig_functions_props",   "kc_trig_eq_ineq",            0.8),
    ("kc_interval_method",        "kc_trig_eq_ineq",            0.6),
    ("kc_sequence_concept",       "kc_limit_concept",           0.7),
    ("kc_function_properties",    "kc_limit_concept",           0.7),
    ("kc_limit_concept",          "kc_derivative_concept",      0.9),
    ("kc_function_properties",    "kc_derivative_concept",      0.8),
    ("kc_derivative_concept",     "kc_derivative_rules",        0.9),
    ("kc_trig_functions_props",   "kc_derivative_rules",        0.7),
    ("kc_derivative_rules",       "kc_tangent_equation",        0.9),
    ("kc_linear_function",        "kc_tangent_equation",        0.7),
    ("kc_derivative_rules",       "kc_derivative_analysis",     0.9),
    ("kc_quadratic_func_graph",   "kc_derivative_analysis",     0.7),
    ("kc_derivative_analysis",    "kc_extremum_values",         0.9),
    ("kc_function_properties",    "kc_extremum_values",         0.7),

    # ================================================================
    # Геометрия 10 класс
    # ================================================================
    ("kc_quadrilaterals",         "kc_polyhedron",              0.8),
    ("kc_area_parallelogram",     "kc_polyhedron",              0.7),
    ("kc_polyhedron",             "kc_regular_polyhedron",      0.8),
    ("kc_triangle_similarity",    "kc_regular_polyhedron",      0.6),
    ("kc_parallel_lines",         "kc_space_lines_planes",      0.8),
    ("kc_quadrilaterals",         "kc_space_lines_planes",      0.6),
    ("kc_space_lines_planes",     "kc_perpendicularity_3d",     0.8),
    ("kc_right_angle",            "kc_perpendicularity_3d",     0.7),
    ("kc_vectors",                "kc_vectors_3d",              0.9),
    ("kc_coordinate_method",      "kc_vectors_3d",              0.8),
    ("kc_vectors_3d",             "kc_space_lines_planes",      0.6),

    # ================================================================
    # Алгебра 11 класс
    # ================================================================
    ("kc_exponential_eq",         "kc_logarithm_concept",       0.9),
    ("kc_power_function_nat",     "kc_logarithm_concept",       0.7),
    ("kc_logarithm_concept",      "kc_logarithm_function",      0.9),
    ("kc_inverse_function",       "kc_logarithm_function",      0.7),
    ("kc_logarithm_function",     "kc_logarithmic_eq",          0.9),
    ("kc_exponential_eq",         "kc_logarithmic_eq",          0.7),
    ("kc_logarithmic_eq",         "kc_logarithmic_ineq",        0.9),
    ("kc_exponential_ineq",       "kc_logarithmic_ineq",        0.7),
    ("kc_quadratic_eq",           "kc_higher_degree_eq",        0.8),
    ("kc_factoring_formulas",     "kc_higher_degree_eq",        0.7),
    ("kc_derivative_analysis",    "kc_antiderivative",          0.9),
    ("kc_derivative_concept",     "kc_antiderivative",          0.8),
    ("kc_antiderivative",         "kc_definite_integral",       0.9),
    ("kc_area_triangle",          "kc_definite_integral",       0.5),
    ("kc_combinatorics_basic",    "kc_permutations_combinations",0.9),
    ("kc_permutations_combinations","kc_binomial_theorem",      0.9),
    ("kc_powers_natural",         "kc_binomial_theorem",        0.7),
    ("kc_logarithmic_eq",         "kc_parametric_problems",     0.7),
    ("kc_trig_eq_ineq",           "kc_parametric_problems",     0.7),
    ("kc_higher_degree_eq",       "kc_parametric_problems",     0.6),

    # ================================================================
    # Геометрия 11 класс
    # ================================================================
    ("kc_polyhedron",             "kc_cylinder_cone",           0.8),
    ("kc_circle_length_area",     "kc_cylinder_cone",           0.7),
    ("kc_cylinder_cone",          "kc_sphere",                  0.8),
    ("kc_circle_basics",          "kc_sphere",                  0.6),
    ("kc_cylinder_cone",          "kc_volumes_solids",          0.9),
    ("kc_sphere",                 "kc_volumes_solids",          0.8),
    ("kc_polyhedron",             "kc_volumes_solids",          0.7),
    ("kc_area_parallelogram",     "kc_volumes_solids",          0.6),
    ("kc_pythagorean_know",       "kc_volumes_solids",          0.6),
]


# ---------------------------------------------------------------------------
# Derived lookups (built once at import time)
# ---------------------------------------------------------------------------

KC_NAMES: dict[str, str] = {kc["kc_id"]: kc["name"] for kc in KCS}
KC_INTRO_GRADE: dict[str, int] = {kc["kc_id"]: kc["grade_introduced"] for kc in KCS}
KC_SUBJECTS: dict[str, str] = {kc["kc_id"]: kc["subject"] for kc in KCS}
KC_DIFFICULTY_BASE: dict[str, float] = {kc["kc_id"]: kc["difficulty_base"] for kc in KCS}
ALL_KC_IDS: list[str] = [kc["kc_id"] for kc in KCS]

SUBJECT_RU: dict[str, str] = {
    "arithmetic": "арифметика",
    "algebra": "алгебра",
    "geometry": "геометрия",
    "statistics": "вероятность",
}

KC_GRAPH: dict[str, list[str]] = {kc_id: [] for kc_id in ALL_KC_IDS}
EDGE_STRENGTHS: dict[tuple[str, str], float] = {}
for _from, _to, _strength in EDGES:
    if _from not in KC_GRAPH[_to]:
        KC_GRAPH[_to].append(_from)
    EDGE_STRENGTHS[(_from, _to)] = _strength
