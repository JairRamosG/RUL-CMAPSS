# Conjunto de Datos de Turbinas de Avión (CMAPSS)

## Descripción General

Este conjunto de datos contiene simulaciones de fallas en motores de turbinas de avión, desarrollado para la competencia de Predicción de Vida Útil Remanente (RUL - Remaining Useful Life). Los datos fueron generados utilizando el modelo de simulación de degradación de motores C-MAPSS (Commercial Modular Aero-Propulsion System Simulation).

**Referencia:** A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

---

## Estructura de los Datos

Los conjuntos de datos consisten en **múltiples series temporales multivariantes**. Cada serie temporal corresponde a un motor diferente de la misma flota. Los datos se dividen en subconjuntos de **entrenamiento** y **prueba**.

### Características de los Motores

- Cada motor comienza con **diferentes grados de desgaste inicial** y variaciones de fabricación (desconocidas para el usuario)
- Este desgaste y variación son considerados **normales** (no son condiciones de falla)
- El motor opera normalmente al inicio de cada serie temporal
- Se desarrolla una falla en algún punto durante la serie
- En el conjunto de entrenamiento, la falla crece hasta la **falla total del sistema**
- En el conjunto de prueba, la serie termina **antes** de la falla del sistema

### Condiciones Operativas

El motor opera bajo **tres configuraciones operativas** que afectan significativamente el rendimiento:
- Configuración Operativa 1
- Configuración Operativa 2
- Configuración Operativa 3

**Nota:** Los datos están contaminados con **ruido de sensores**.

---

## Conjuntos de Datos Disponibles

### FD001
| Característica | Valor |
|----------------|-------|
| **Trayectorias de Entrenamiento** | 100 |
| **Trayectorias de Prueba** | 100 |
| **Condiciones** | UNA (Nivel del Mar) |
| **Modos de Falla** | UNO (Degradación de HPC) |

### FD002
| Característica | Valor |
|----------------|-------|
| **Trayectorias de Entrenamiento** | 260 |
| **Trayectorias de Prueba** | 259 |
| **Condiciones** | SEIS |
| **Modos de Falla** | UNO (Degradación de HPC) |

### FD003
| Característica | Valor |
|----------------|-------|
| **Trayectorias de Entrenamiento** | 100 |
| **Trayectorias de Prueba** | 100 |
| **Condiciones** | UNA (Nivel del Mar) |
| **Modos de Falla** | DOS (Degradación de HPC, Degradación del Ventilador) |

### FD004
| Característica | Valor |
|----------------|-------|
| **Trayectorias de Entrenamiento** | 248 |
| **Trayectorias de Prueba** | 249 |
| **Condiciones** | SEIS |
| **Modos de Falla** | DOS (Degradación de HPC, Degradación del Ventilador) |

---

## Objetivo del Problema

El objetivo de la competencia es **predecir el número de ciclos operativos remanentes antes de la falla** en el conjunto de prueba, es decir:

> **RUL (Remaining Useful Life)** = Número de ciclos operativos después del último ciclo registrado que el motor continuará operando

Se proporciona un vector con los **valores reales de RUL** para los datos de prueba.

---

## Formato de los Archivos

Los datos se proporcionan como **archivos de texto comprimidos en ZIP** con **26 columnas** de números separados por espacios.

### Estructura de Columnas

| Columna | Descripción |
|---------|-------------|
| 1 | Número de unidad |
| 2 | Tiempo (en ciclos) |
| 3 | Configuración operativa 1 |
| 4 | Configuración operativa 2 |
| 5 | Configuración operativa 3 |
| 6 | Medición de sensor 1 |
| 7 | Medición de sensor 2 |
| ... | ... |
| 26 | Medición de sensor 26 |

### Descripción Detallada de Columnas

```plaintext
1)  unit number
2)  time, in cycles
3)  operational setting 1
4)  operational setting 2
5)  operational setting 3
6)  sensor measurement  1
7)  sensor measurement  2
...
26) sensor measurement  26