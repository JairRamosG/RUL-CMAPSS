# ESPECIFICACIÓN TÉCNICA Y GUÍA DE EJECUCIÓN DETALLADA: ISSUE #3

**Título del Issue:** `[FD001] Análisis Exploratorio de Datos (EDA) Orientado a Diagnóstico y Pronóstico de RUL`  
**Asignado a:** `2_feature_engineer` / `1_research_planner`  
**Dataset Objetivo:** C-MAPSS Subconjunto `FD001` (`datos/train_FD001.txt`, `datos/test_FD001.txt`, `datos/RUL_FD001.txt`)  
**Estatus:** Ready for Development / Complete Specification  

---

## 1. OBJETIVO GENERAL Y CONTEXTO ACADÉMICO

### 1.1 Goal (Objetivo)
Ejecutar un Análisis Exploratorio de Datos (EDA) formal, cuantitativo y reproducible sobre el subconjunto `FD001` del dataset C-MAPSS, el cual debe ser desarrollado dentro del Jupyter Notebook `notebooks/01_eda_FD001.ipynb`. 

El propósito central de este EDA no es realizar una descripción estadística convencional o superficial, sino establecer la **fundamentación metodológica formal** de las siguientes decisiones críticas dentro del pipeline de datos:

1. Inspección de la integridad y estructura de las trayectorias de falla (*run-to-failure*).
2. Selección y reducción de variables de entrada (descarte cuantitativo de sensores constantes).
3. Fundamentación del etiquetado de la variable objetivo mediante el modelo *Piecewise Linear RUL* con saturación máxima en $RUL_{max} = 125$ ciclos.
4. Evaluación cuantitativa de la tendencia monotónica de los sensores seleccionados mediante el coeficiente de correlación de Spearman.
5. Definición de la dimensión de la ventana temporal de observación ($W = 30$ lecturas pasadas) y análisis de la densidad de probabilidad (KDE) para justificar la cuantización/binning requerida por la Memoria Asociativa.

Cada decisión tomada dentro del código y la documentación debe estar explícitamente respaldada en el estado del arte consolidado mediante citas bibliográficas estándar.

---

## 2. CRITERIOS DE ACEPTACIÓN (ACCEPTANCE CRITERIA)

El agente o desarrollador que ejecute este issue deberá validar el cumplimiento de los siguientes puntos:

- [ ] **Sección 1 (Estructura e Integridad):** Se analiza el archivo `train_FD001.txt`, verificando la ausencia de valores nulos o infinitos y calculando el número total de unidades/motores (100 unidades), así como las estadísticas de vida útil: ciclos mínimos, máximos, promedio y desviación estándar por motor.
- [ ] **Sección 2 (Filtrado por Varianza):** Se calcula la desviación estándar de los 21 sensores en todo el dataset de entrenamiento. Se eliminan cuantitativamente los 7 sensores con varianza nula (`sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`) y se conserva explícitamente el subconjunto estándar de 14 sensores informativos (`sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21`).
- [ ] **Sección 3 (Degradación y Piecewise Linear RUL):** Se implementa el cálculo del RUL lineal y el RUL truncado (*Piecewise Linear*) fijando un límite superior de $RUL_{max} = 125$ ciclos. Se genera un gráfico de dos ejes Y donde se compare la trayectoria de un sensor representativo (ej. `sensor_11`) contra la curva de RUL truncado para la unidad #1.
- [ ] **Sección 4 (Monotonicidad de Spearman):** Se calcula la matriz/tabla de correlación de Spearman entre los 14 sensores retenidos y la etiqueta `RUL_piecewise`. Se ordenan las correlaciones de forma descendente y se genera una gráfica de barras horizontal/vertical identificando los sensores con tendencias de degradación positiva y negativa.
- [ ] **Sección 5 (Estructura para Memoria Asociativa):** Se documenta la ventana temporal $W = 30$ lecturas. Se realiza un análisis de Estimación de Densidad de Kernel (KDE) sobre sensores clave para justificar visualmente si la distribución requiere una cuantización por cuantiles (*Quantile Binning*) o por intervalos uniformes (*Uniform Binning*) para el posterior proceso de binarización/codificación.
- [ ] **Referencias Bibliográficas Incorporadas:** Se colocan comentarios inline en cada bloque del código y en las celdas de Markdown del notebook citando el autor, año y concepto aplicado.
- [ ] **Exportación de Artefactos:** Todas las gráficas resultantes deben guardarse automáticamente en la ruta `reports/figures/` con una resolución mínima de 300 DPI y en formato PNG.
- [ ] **Restricción de Estilo Académico:** Queda estrictamente prohibido el uso de emojis o emoticones en las celdas de Markdown, comentarios de código, salidas impresas o mensajes de confirmación.

---

## 3. FUERA DE ALCANCE (OUT OF SCOPE) Y RESTRICCIONES

### Fuera de Alcance:
1. Entrenamiento, validación o evaluación de modelos de aprendizaje automático (Memoria Asociativa, Redes Neuronales, Random Forest, Baselines).
2. Normalización por regímenes de operación mediante clustering K-Means sobre las variables `setting_1`, `setting_2` y `setting_3` (esta técnica es exclusiva de los subconjuntos multirégimen `FD002` y `FD004`).
3. Modificación, sobrescritura o reestructuración de los archivos planos de texto ubicados en la carpeta `datos/`.

### Restricciones Técnicas:
* **Lenguaje:** Python 3.10 o superior.
* **Librerías Permitidas:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`.
* **Rigor Científico:** El ajuste o fit de transformaciones (ej. cálculo de varianza, promedios o cuantiles) debe calcularse de manera exclusiva sobre el conjunto de entrenamiento (`train_FD001.txt`) para prevenir la fuga de información (*data leakage*).

---

## 4. ESTRUCTURA METODOLÓGICA DEL EXPERIMENTO EDA

```mermaid
graph TD
    A[Inicio: Archivos de Datos C-MAPSS FD001] --> B[Sección 1: Inspección de Estructura e Integridad]
    B --> C[Lectura de train_FD001.txt, test_FD001.txt, RUL_FD001.txt]
    C --> D[Cálculo de Estadísticas de Trayectoria por Motor]
    
    D --> E[Sección 2: Filtrado Cuantitativo de Sensores]
    E --> F[Cálculo de Desviación Estándar y Varianza de 21 Sensores]
    F --> G{¿Varianza > 0?}
    G -- No: std == 0 --> H[Eliminar 7 Sensores Constantes: s1, s5, s10, s16, s18, s19]
    G -- Sí: std > 0 --> I[Retener Subconjunto Estándar de 14 Sensores]
    
    H --> J[Sección 3: Justificación de RUL_max = 125]
    I --> J
    J --> K[Cálculo de RUL Lineal Real por Ciclo]
    K --> L[Aplicar Límite de Saturación Piecewise: min RUL, 125]
    
    L --> M[Sección 4: Evaluación de Monotonicidad]
    M --> N[Matriz de Correlación de Spearman: Sensores vs. RUL]
    N --> O[Identificación de Tendencias Monotónicas Positivas y Negativas]
    
    O --> P[Sección 5: Dinámica Temporal y Espacio de Entradas]
    P --> Q[Definición de Ventana Temporal W = 30 Lecturas]
    Q --> R[Análisis de Densidad KDE para Cuantización de Memoria Asociativa]
    R --> S[Generación de Reportes Visuales en reports/figures/]
    S --> T[Fin: Fundamentación del Pipeline Completa]

## 5. ESPECIFICACIÓN DETALLADA PASO A PASO (GUÍA DE CÓDIGO Y REFERENCIAS)

A continuación se desglosa cómo debe construirse cada bloque de código y la documentación dentro del notebook `notebooks/01_eda_FD001.ipynb`.


### 5.1 Configuración Inicial e Importación de Módulos

**Qué se debe hacer:**
- Configurar las rutas de importación, las opciones de despliegue de pandas y el estilo global de matplotlib asegurando tipografía de tipo serif, tamaños de fuente legibles para publicación y resolución de 300 DPI.
- Crear el directorio `reports/figures/` utilizando el módulo `os`.

**Estructura del código sugerida:**
- Usar `pandas.read_csv` con el separador de espacio en blanco variable `sep=r'\s+'` y asignación explícita de los nombres de columnas oficiales:
  - Columnas de Índice: `['unit_number', 'time_cycles']`
  - Columnas de Regímenes: `['setting_1', 'setting_2', 'setting_3']`
  - Columnas de Sensores: `['sensor_1', ..., 'sensor_21']`

**Referencia Bibliográfica Aplicada:**
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation*. In Proceedings of the 1st International Conference on Prognostics and Health Management (PHM '08), pp. 1-9. IEEE.
- **Sustento:** Publicación inicial que establece la estructura de archivos planos, la denominación de variables y el marco de simulación del benchmark C-MAPSS.


### 5.2 Sección 1: Inspección de Estructura e Integridad del Dataset

**Qué se debe hacer:**
- Cargar el dataframe de entrenamiento `df_train` leyendo el archivo `../datos/train_FD001.txt`.
- Verificar la ausencia de datos faltantes evaluando `df_train.isnull().sum().sum()`.
- Calcular la vida útil máxima (*run-to-failure*) por cada motor individual mediante agrupamiento: `df_train.groupby('unit_number')['time_cycles'].max()`.
- Imprimir en pantalla las estadísticas descriptivas de la vida útil: ciclo mínimo, máximo, promedio y mediana.
- Generar un histograma con estimación de densidad kernel (`sns.histplot` con `kde=True`) que muestre la distribución de la longevidad de las 100 unidades de entrenamiento.
- Guardar la figura en `../reports/figures/01_distribucion_vida_util_FD001.png`.

**Referencia Bibliográfica Aplicada:**
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation*. IEEE.
- **Sustento:** Justifica la naturaleza heterogénea de las trayectorias de falla en los motores a reacción, donde cada unidad inicia con grados de desgaste iniciales variados pero dentro de especificaciones operativas normales.


### 5.3 Sección 2: Filtrado Cuantitativo de Sensores por Varianza Nula

**Qué se debe hacer:**
- Calcular la desviación estándar (`std()`) o la varianza (`var()`) de los 21 sensores en `df_train[SENSOR_COLS]`.
- Programar una condición lógica para identificar y almacenar en una lista los sensores cuya desviación estándar sea exactamente $0.0$.
- Demostrar cuantitativamente que los sensores `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18` y `sensor_19` presentan valores constantes a lo largo de toda la simulación en FD001.
- Filtrar el dataframe para conservar únicamente los 14 sensores con varianza mayor a cero (`KEPT_SENSORS`).
- Incluir una aserción condicional (`assert len(KEPT_SENSORS) == 14`) para garantizar la integridad del proceso de selección de variables.

**Referencia Bibliográfica Aplicada:**
- Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). *Long Short-Term Memory Network for Remaining Useful Life Estimation*. In IEEE International Conference on Big Data, pp. 1572-1578. IEEE.
- **Sustento:** Establece la metodología estándar de reducción del espacio de características a 14 sensores en C-MAPSS, reduciendo la dimensión de entrada sin sacrificar información diagnóstica ni capacidad predictiva.


### 5.4 Sección 3: Visualización de Degradación y Justificación de $RUL_{max} = 125$

**Qué se debe hacer:**
- Calcular el RUL lineal teórico por ciclo: para cada motor, restar el ciclo actual del ciclo máximo alcanzado por dicha unidad ($RUL_{linear} = TotalCycles - CurrentCycle$).
- Implementar la función de saturación Piecewise Linear limitando los valores de RUL a un techo máximo de $125$ ciclos usando `df_train['RUL_linear'].clip(upper=125)`.
- Seleccionar la trayectoria de la unidad #1 (`unit_number == 1`).
- Construir una figura gráfica de doble eje Y usando `fig, ax1 = plt.subplots()` y `ax1.twinx()`:
  - En el eje Y primario (izquierdo): Graficar la curva del `sensor_11` (Temperatura a la salida del compresor de alta presión, HPC) a lo largo de los ciclos temporales.
  - En el eje Y secundario (derecho): Graficar la curva del `RUL_piecewise` en línea discontinua color negro.
- Guardar el gráfico en `../reports/figures/02_perfil_degradacion_piecewise_FD001.png`.

**Referencia Bibliográfica Aplicada:**
- Heimes, F. O. (2008). *Recurrent neural networks for remaining useful life estimation*. In Proceedings of the 1st International Conference on Prognostics and Health Management (PHM '08), pp. 1-6. IEEE.
- **Sustento:** Primer trabajo en proponer la función de etiquetado Piecewise Linear RUL truncada a 125 ciclos. Demuestra que durante las etapas iniciales de operación el sistema se encuentra en estado saludable y los sensores no muestran patrones de degradación distinguibles del ruido.


### 5.5 Sección 4: Matriz de Correlación de Monotonicidad de Spearman

**Qué se debe hacer:**
- Calcular el coeficiente de correlación de rangos de Spearman (`scipy.stats.spearmanr`) entre cada uno de los 14 sensores retenidos y la variable objetivo truncada `RUL_piecewise`.
- Estructurar las correlaciones obtenidas dentro de un `pandas.DataFrame` ordenado en forma descendente según el valor absoluto o relativo de la correlación.
- Clasificar visual y cuantitativamente los sensores en dos grupos:
  - **Sensores con correlación negativa fuerte:** Las lecturas aumentan conforme el RUL disminuye/el motor se degrada (ej. `sensor_2`, `sensor_3`, `sensor_4`, `sensor_11`, `sensor_15`, `sensor_17`).
  - **Sensores con correlación positiva fuerte:** Las lecturas disminuyen conforme el motor se degrada (ej. `sensor_7`, `sensor_11`, `sensor_12`, `sensor_20`, `sensor_21`).
- Generar un gráfico de barras utilizando `seaborn.barplot` para visualizar la fuerza de la relación monotónica de cada sensor.
- Guardar en `../reports/figures/03_correlacion_spearman_sensores_FD001.png`.

**Referencia Bibliográfica Aplicada:**
- Coble, J. B., & Hines, R. E. (2009). *Identifying optimal prognostic parameters for maintenance decision making*. In Proceedings of the Annual Conference of the PHM Society (Vol. 1, pp. 1-12).
- **Sustento:** Fundamenta la monotonicidad como una propiedad deseable e indispensable en los parámetros prognósticos (Health Indicators), justificando el uso de Spearman sobre Pearson al no asumir una relación lineal estricta.


### 5.6 Sección 5: Dinámica Temporal y Análisis de Densidad para Memoria Asociativa

**Qué se debe hacer:**
- Definir formalmente la dimensión de la ventana temporal móvil $W = 30$ lecturas pasadas, la cual estructurará las matrices de entrada de dimensión $(W, 14)$ para la Memoria Asociativa.
- Calcular y graficar la Estimación de Densidad de Kernel (`sns.kdeplot`) sobre un grupo de sensores clave (ej. `sensor_2`, `sensor_4`, `sensor_7`, `sensor_11`).
- Analizar la simetría y las colas de las distribuciones para argumentar si el posterior proceso de binarización/discretización (*binning*) requerido por la Memoria Asociativa debe ejecutarse mediante intervalos de ancho uniforme (*Uniform Binning*) o intervalos basados en cuantiles (*Quantile Binning*).
- Guardar la gráfica de densidad resultante en `../reports/figures/04_distribucion_kde_sensores_FD001.png`.

**Referencia Bibliográfica Aplicada:**
- Li, X., Ding, Q., & Sun, J. (2018). *Remaining useful life estimation in prognostics using deep convolution neural networks*. Reliability Engineering & System Safety, 172, 1-11.
- **Sustento:** Valida que una ventana de tiempo $W = 30$ captura de manera óptima la memoria temporal y la dinámica de degradación en C-MAPSS sin sobrecargar el espacio muestral.


---

## 6. GUÍA DE EXTENSIÓN PARA SUBCONJUNTOS FUTUROS (FD002, FD003, FD004)

Este desarrollo sobre FD001 servirá como la plantilla base. Para los siguientes subconjuntos, el código deberá extenderse integrando las siguientes etapas adicionales:

- **Subconjunto FD002 (6 Regímenes de Operación, 1 Modo de Falla):**
  *Adición requerida:* Un análisis de clustering K-Means sobre `setting_1`, `setting_2` y `setting_3` para identificar los 6 regímenes de carga y aplicar normalización Z-score condicional por cluster (Referencia: Heimes, 2008; Peel, 2008).

- **Subconjunto FD003 (1 Régimen de Operación, 2 Modos de Falla: HPC y Fan):**
  *Adición requerida:* Un análisis comparativo de la firma de degradación entre grupos de motores para distinguir el patrón de falla en el Fan respecto al patrón de falla en el Compresor de Alta Presión (Referencia: Saxena et al., 2008).

- **Subconjunto FD004 (6 Regímenes de Operación, 2 Modos de Falla):**
  *Adición requerida:* La combinación integrada del clustering por régimen de operación con la separación por modo de falla.


---

## 7. RESUMEN DE REFERENCIAS BIBLIOGRÁFICAS PARA CITA EN TESIS

A continuación se reúne la lista bibliográfica consolidada para su posterior uso en la redacción del marco metodológico de la tesis:

- **Coble, J. B., & Hines, R. E. (2009).** *Identifying optimal prognostic parameters for maintenance decision making*. In Proceedings of the Annual Conference of the Prognostics and Health Management Society (Vol. 1, pp. 1-12).
- **Heimes, F. O. (2008).** *Recurrent neural networks for remaining useful life estimation*. In Proceedings of the 1st International Conference on Prognostics and Health Management (PHM '08), pp. 1-6. IEEE.
- **Li, X., Ding, Q., & Sun, J. (2018).** *Remaining useful life estimation in prognostics using deep convolution neural networks*. Reliability Engineering & System Safety, 172, 1-11.
- **Peel, L. (2008).** *Data driven prognostics using a Kalman filter ensemble of neural network models*. In 2008 International Conference on Prognostics and Health Management, pp. 1-6. IEEE.
- **Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).** *Damage propagation modeling for aircraft engine run-to-failure simulation*. In Proceedings of the 1st International Conference on Prognostics and Health Management (PHM '08), pp. 1-9. IEEE.
- **Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017).** *Long Short-Term Memory Network for Remaining Useful Life Estimation*. In IEEE International Conference on Big Data (Big Data), pp. 1572-1578. IEEE.
