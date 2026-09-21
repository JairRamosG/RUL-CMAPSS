"""
Módulo de Selección de Características para RUL-CMAPSS.

Implementa un enfoque híbrido en dos etapas:
1. Filtro univariado (Mutual Information) para descartar ruido.
2. Selector embebido multivariado (Random Forest) para capturar interacciones.
Compatible con Scikit-learn Pipeline y protocolo anti-leakage.
"""
