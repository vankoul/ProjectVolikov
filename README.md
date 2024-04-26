# ProjectVolikov
Итоговое задание по дисциплине **Проект - применение машинного обучения в креативных индустриях**

**Выполнил:** Воликов Иван Дмитриевич группы **МИК21**

**Задание:** Анализ данных по методологии CrispDM
**Цель:** Разработать навыки работы с данными в ходе выполнения проекта, основанного на применении машинного обучения в креативных индустриях.


**Описание задания:** Решить задачу с винами (www.kaggle.com...ne-quality), оформив в виде CrispDM-подхода.

**Цель:** Классифицировать португальские вина «Vinho Verde» как "плохие", "нормальные" или "отличные"

**Описание**
Данные были загружены из репозитория машинного обучения UCI.
Два набора данных относятся к красному и белому вариантам португальского вина «Vinho Verde». Подробнее см. в ссылке [Cortez et al., 2009]. Из-за проблем с конфиденциальностью и логистикой доступны только физико-химические (входные) и органолептические (выходные) переменные (например, нет данных о сортах винограда, марке вина, продажной цене вина и т. д.).
Эти наборы данных можно рассматривать как задачи классификации или регрессии. Классы упорядочены и не сбалансированы (например, нормальных вин гораздо больше, чем отличных или плохих). Алгоритмы обнаружения выбросов можно использовать для определения нескольких отличных или плохих вин.
Описание от Kaggle

**Импорт библиотек**
import pandas as pd
import numpy as np

Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor

Modelling Helpers
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import  Normalizer , scale
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from pprint import pprint

Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

Configure visualisations
%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


**Описание переменных (по данным физико-химических тестов):**
	count	mean	std	min	25%	50%	75%	max
fixed acidity	6487.0	7.216579	1.296750	3.80000	6.40000	7.00000	7.70000	15.90000

volatile acidity	6489.0	0.339691	0.164649	0.08000	0.23000	0.29000	0.40000	1.58000

citric acid	6494.0	0.318722	0.145265	0.00000	0.25000	0.31000	0.39000	1.66000

residual sugar	6495.0	5.444326	4.758125	0.60000	1.80000	3.00000	8.10000	65.80000

chlorides	6495.0	0.056042	0.035036	0.00900	0.03800	0.04700	0.06500	0.61100

free sulfur dioxide	6497.0	30.525319	17.749400	1.00000	17.00000	29.00000	41.00000	289.00000

total sulfur dioxide	6497.0	115.744574	56.521855	6.00000	77.00000	118.00000	156.00000	440.00000

density	6497.0	0.994697	0.002999	0.98711	0.99234	0.99489	0.99699	1.03898

pH	6488.0	3.218395	0.160748	2.72000	3.11000	3.21000	3.32000	4.01000

sulphates	6493.0	0.531215	0.148814	0.22000	0.43000	0.51000	0.60000	2.00000

alcohol	6497.0	10.491801	1.192712	8.00000	9.50000	10.30000	11.30000	14.90000

quality	6497.0	5.818378	0.873255	3.00000	5.00000	6.00000	6.00000	9.0000
0
1 - fixed acidity (большинство кислот, связанных с вином, либо фиксированные, либо нелетучие (легко не испаряются))

2 - volatile acidity (количество уксусной кислоты в вине, которая при слишком высоком уровне может привести к неприятному "уксусному" вкусу)

3 - citric acid (Лимонная кислота, содержащаяся в небольших количествах, может придать вину «свежесть» и аромат)

4 - residual sugar (количество сахара, оставшееся после прекращения брожения, редко можно найти вина с содержанием сахара менее 1 г/л, а вина с содержанием сахара более 45 г/л считаются сладкими)

5 - chlorides (количество соли в вине)

6 - free sulfur dioxide (свободная форма SO^2 (диоксида серы) существует в равновесии между молекулярным SO^2 (в виде растворенного газа) и бисульфит-ионом; предотвращает рост микробов и окисление вина)

7 - total sulfur dioxide (количество свободных и связанных форм диоксида серы; при низких концентрациях диоксида серы практически не обнаруживается в вине, но при концентрациях свободного диоксида серы более 50 частей на миллион диоксида серы становится заметным в аромате и вкусе вина)

8 - density (плотность воды близка к плотности воды в зависимости от процентного содержания спирта и сахара)

9 - pH (описывает, насколько кислым или щелочным является вино по шкале от 0 (очень кислое) до 14 (очень щелочное); большинство вин имеют рН от 3 до 4 по шкале pH)

10 - sulphates (добавка к вину, которая может способствовать повышению уровня газообразного диоксида серы (SO2), действующая как антимикробное и антиоксидантное средство)

11 - alcohol (процент содержания алкоголя в вине) Целевая переменная (по данным сенсорной оценки): 12 - quality (целевая переменная (на основе сенсорных данных, оценка от 0 до 10))

(Больше информации на сайте Kaggle)

**Оценка результатов данных (после проверки)**
[ ]
accuracy_model1 = (1 - best_rmse_model1)

accuracy_model2 = (1 - best_rmse_model2)

print('Accuracy according to RMSE for Random Forest',100*accuracy_model1)

print('Accuracy according to RMSE for Random Forest using the results of GridSearchCV',100*accuracy_model2)


**Accuracy according to RMSE for Random Forest 89.54798685264807

Accuracy according to RMSE for Random Forest using the results of GridSearchCV 89.4579790575403**
Таким образом, точность первой модели немного больше, чем второй
