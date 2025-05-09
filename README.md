####Tarea 1: Familiarizarse con el conjunto de datos
import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
print("Primeras filas:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())
print("\nEstadísticas básicas:")
print(df.describe())
for col in df.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()
###Tarea 2: Generar estadísticas descriptivas y visualizaciones
# en esta grafica se puede on+bservar como los datos sobre las viviendas en boston combian dependiendo de su obicacion y su antiguedad}

####Para el "Valor medio de las viviendas ocupadas por sus propietarios" proporcione un diagrama de caja (boxplot)
import pandas as pd
import matplotlib.pyplot as plt

boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df = pd.read_csv(boston_url)
plt.figure(figsize=(6, 4))
plt.boxplot(boston_df["MEDV"])
plt.title("Diagrama de caja del valor medio de las viviendas (MEDV)")
plt.ylabel("Valor en miles de dólares")
plt.grid(True)
plt.show()

####Proporcione un diagrama de barras para la variable "río Charles
chas_counts = boston_df["CHAS"].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(chas_counts.index.astype(str), chas_counts.values, color="lightgreen", edgecolor="black")
plt.title("Número de viviendas según proximidad al río Charles (CHAS)")
plt.xlabel("CHAS (1 = cerca del río, 0 = no cerca)")
plt.ylabel("Frecuencia")
plt.grid(True, axis='y')
plt.show()


####Proporcione un boxplot para la variable MEDV frente a la variable EDAD. (Discretice la variable edad en tres
# grupos de 35 años o menos, entre 35 y 70 años y 70 años o más)
def clasificar_edad(edad):
    if edad <= 35:
        return "0-35 años"
    elif edad <= 70:
        return "36-70 años"
    else:
        return "71+ años"
boston_df["grupo_edad"] = boston_df["AGE"].apply(clasificar_edad)
plt.figure(figsize=(6, 4))
boston_df.boxplot(column="MEDV", by="grupo_edad", grid=False)
plt.title("Valor medio de vivienda (MEDV) por grupo de edad de la propiedad")
plt.suptitle("")  # Eliminar título automático
plt.xlabel("Grupo de edad de la propiedad")
plt.ylabel("Valor medio (miles de $)")
plt.show()

####Proporcione un diagrama de dispersión para mostrar la relación entre las concentraciones de óxido nítrico y
# la proporción de acres comerciales no minoristas por ciudad. ¿Qué puede decir sobre la relación?
plt.figure(figsize=(6, 4))
plt.scatter(boston_df["INDUS"], boston_df["NOX"], alpha=0.7, color="coral", edgecolors="k")
plt.title("Relación entre actividad industrial (INDUS) y contaminación (NOX)")
plt.xlabel("Proporción de acres industriales (INDUS)")
plt.ylabel("Concentración de óxidos nítricos (NOX)")
plt.grid(True)
plt.show()
####Cree un histograma para la variable proporción de alumnos por profesor
plt.figure(figsize=(6, 4))
plt.hist(boston_df["PTRATIO"], bins=15, color="skyblue", edgecolor="black")
plt.title("Distribución de la proporción alumnos/profesor (PTRATIO)")
plt.xlabel("PTRATIO (alumnos por profesor)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.show()

###Tarea 3: Utilice las pruebas adecuadas para responder a las preguntas que se le plantean

####prueba T
from scipy.stats import ttest_ind

near_river = boston_df[boston_df["CHAS"] == 1]["MEDV"]
away_river = boston_df[boston_df["CHAS"] == 0]["MEDV"]
t_stat, p_value = ttest_ind(near_river, away_river, equal_var=False)
print(f"p-valor: {p_value:.4f}")

###ANOVA
from scipy.stats import f_oneway

group_1 = boston_df[boston_df["grupo_edad"] == "0-35 años"]["MEDV"]
group_2 = boston_df[boston_df["grupo_edad"] == "36-70 años"]["MEDV"]
group_3 = boston_df[boston_df["grupo_edad"] == "71+ años"]["MEDV"]
f_stat, p_value = f_oneway(group_1, group_2, group_3)
print(f"p-valor: {p_value:.4f}")

###correlacion de personas con NOX y INDUS
from scipy.stats import pearsonr

corr, p_value = pearsonr(boston_df["NOX"], boston_df["INDUS"])
print(f"Coeficiente de correlación: {corr:.3f}, p-valor: {p_value:.4f}")

###registro lineal
import statsmodels.api as sm

X = sm.add_constant(boston_df["DIS"])  # Variable independiente
y = boston_df["MEDV"]                  # Variable dependiente
model = sm.OLS(y, X).fit()
print(model.summary())

