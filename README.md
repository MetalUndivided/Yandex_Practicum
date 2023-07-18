# Yandex_Practicum

Проекты, выполненные в процессе обучения в Яндекс Практикум по программе "Специалист по Data Science".

_____

## Список проектов

Проекты расположены в хронологическом порядке от более старых к более новым

1. [Исследование надежности заемщиков](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/01_Data_Preprocessing)  - на основании данных о заемщиках сделаны качественные выводы о том, какие признаки заемщиков влияют на их надежность. Проведена предобработка данных, а так же простой EDA при помощи сводных таблиц. Использованные интсрументы - pandas.

2. [Определение стоимости недвижимости](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/02_Exploratory_Analysis) - используя данные сервиса Яндекс.Недвижимость об объектах в Санкт-Петербурге и Ленинградской области, определены параметры, наиболее влияющие на стоимость недвижимости, а так же определены наиболее дорогие населенные пункты. Проведена предобработка данных, проведен EDA. Использованные инструменты - pandas, matplotlib.

3. [Исследование рынка российского кинопроката](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/03_Cumulative_1) - используя данные о российском кинопрокате от Министерства Культуры, сделаны выводы об успешности фильмов, получивших государственную поддержку. Проведен препроцессинг данных, EDA, изучено влияние объема гос. финансирования на характеристики фильмов. Использованные инструменты - pandas, matplotlib.

4. [Модель рекоммендации тарифов](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/05_ML_intro) - на основании данных о клиентах, построена классификационная модель, рекоммендующая им тариф связи. Проведены типичные шаги - препроцессинг данных, EDA, а так же проведено разбиение данных и рассмотрено несколько простых моделей, из которых выбрана лучшая. Использованные инструменты - pandas, matplotlib, sklearn.

5. [Предсказание оттока клиентов банка](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/06_Supervised_ML) - используя данные о клиентах банка, построена классификационная модель, предсказывающая, покинет ли клиент банк или нет. Проведены типичные подготовочные шаги - препроцессинг данных, EDA, разделение данных на выборке. Так же опробованы несколько способов борьбы с дисбалансом классов, обучены несколько моделей, из которых выбрана лучшая. Использованные инструменты - pandas, matplotlib, sklearn.

6. [Выбор локации для скважины](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/07_Business_ML) - на основании данных о точках в 3 перспективных нефтеносных участках, выбран участок с наиболее высокой ожидаемой прибылью и наименьшей вероятностью убыточности. Проведены типичные шаги - препроцессинг, EDA, разделение. Обучена модель линейной регрессии, при помощи bootstrap расчитаны средняя прибыль, доверительный интервал и вероятность убытков. Использованные инструменты - pandas, matplotlib, sklearn, scipy.

7. [Модель эффективности обогащения золотой руды](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/08_Cumulative_2) - используя исторические данные об обогащении золотой руды, построена модель, предсказывающая эффективность обогащения. Проведены обычные шаги - препроцессинг данных и EDA. Для нескольких вариантов моделей проведен подбор гиперпараметров, выбрана лучшая модель. Использованные инструменты - pandas, matplotlib, sklearn.

8. [Защита данных клиентов](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/09_Linear_Algebra) - предложен, обоснован, реализован и проверен алгоритм, который позволяет зашифровать данные так, чтобы шифровка не повлияла на качество модели линейной регрессии. Приведено математическое обоснование сохранения точности модели с точки зрения линейной алгебры. Использованные инструменты - pandas, numpy.

9. [Определение стоимости автомобилей](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/10_Numerical_Methods) - на основании данных о продаваемых автомобилях, построена модель, предсказывающая их стоимость. Проведены стандартные шаги - препроцессинг и EDA. Проведен подбор гиперпараметров для нескольких моделей, в т.ч. модели градиентного бустинга. Проведен анализ скорости обучения, работы и точности моделей, выбрана лучшая. Использованные инструменты - pandas, matplotlib, sklearn, LightGBM.

10. [Прогнозирование заказов такси | Временные ряды](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/11_Time_Series) - используя исторические данные о заказах такси в Московских аэропортах, построена модель, предсказывающая количество заказов такси на следующий час. Проведен препроцессинг данных, изучено влиянеие сезонности на количество заказов. Из временного ряда выделены признаки, обучено несколько моделей с подбором гиперпараметров, выбрана лучшая. Использованные инструменты - pandas, matplotlib, statsmodels, catboost, sklearn.

11. [Определение таксичных комментариев | NLP](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/12_Natural_Language_Processing) - используя размеченный корпус комментариев с Википедии, обучена модель, предсказывающая негативные комментарии. Проведена лемматизация и векторизация корпуса, обучено несколько моделей, выбрана оптимальная. Использованные инструменты - pandas, регулярные выражения, nltk, sklearn.

12. [Определение возраста людей по фото | CV](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/13_Computer_Vision) - на основании размеченного набора фото (APPA-REAL), обучена сверточная нейронная сеть (ResNet50), предсказывающая возраст людей. Проведен EDA, произведена настройка модели. Использованные инструменты - pandas, matplotlib, keras.

13. [Предсказание температуры стали на предприятии](https://github.com/MetalUndivided/Yandex_Practicum/tree/master/14_Final) - используя исторические данные о температуре стали и проведенных над ней опреациях, обучена модель, предсказывающая температуру стали после проведения над ней операций. Проведен обширный препроцессинг данных с мержами из разных источников, EDA. Обучено несколько моделей с подбором гиперпараметров, в т.ч. модель градиентного бустинга. Выбрана наилучшая модель, проведен анализ важности признаков. Использованные инструменты - pandas, matplotlib, sklearn, catboost.

14.  
