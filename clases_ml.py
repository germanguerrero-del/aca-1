"""Clases propias del proyecto (K-Means como clasificador y agente de Aprendizaje por Refuerzo).
Se guardan en un módulo aparte para que Streamlit pueda reconstruir los modelos entrenados."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans


class KMeansClasificador(ClassifierMixin, BaseEstimator):
    """K-Means usado como clasificador: agrupa y asigna a cada grupo la proporción de ventas altas del entrenamiento."""
    def __init__(self, n_clusters=8, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        # Paso 1 del método: agrupar con K-Means
        self.km_ = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit(X)
        etiquetas = self.km_.labels_
        # Paso 2: proporción de ventas altas de cada grupo (el 'nombre' del grupo)
        self.tasa_ = np.array([y[etiquetas == k].mean() if (etiquetas == k).any() else y.mean()
                               for k in range(self.n_clusters)])        # proporción de ventas altas por grupo
        self.classes_ = np.array([0, 1])
        return self

    # Un dato nuevo va a su grupo más cercano y hereda la proporción de ese grupo
    def predict_proba(self, X):
        p = self.tasa_[self.km_.predict(X)]
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class AgenteQLearning(ClassifierMixin, BaseEstimator):
    """Agente de Aprendizaje por Refuerzo (Q-learning tabular de un paso) que decide 'alta' o 'normal'."""
    def __init__(self, episodios=30, epsilon_ini=0.30, epsilon_fin=0.02, min_obs=2,
                 recompensas=(1.0, 1.0, -1.0, -2.0), random_state=42):
        self.episodios = episodios
        self.epsilon_ini = epsilon_ini
        self.epsilon_fin = epsilon_fin
        self.min_obs = min_obs
        self.recompensas = recompensas        # (acierto alta, acierto normal, falsa alarma, alta no detectada)
        self.random_state = random_state

    # Estado = producto (si es frecuente); si no, su subcategoría; si no, 'desconocido'
    def _estado(self, X):
        """Estado = producto (si es conocido y frecuente); si no, su subcategoría; si no, 'desconocido'."""
        out = []
        for p, s in zip(X["Product ID"].to_numpy(), X["Sub-Category"].to_numpy()):
            out.append(self.estados_.get(p, self.estados_.get(s, self.n_estados_ - 1)))
        return np.array(out)

    def fit(self, X, y):
        # Generador de azar con semilla (para explorar de forma reproducible)
        rng = np.random.default_rng(self.random_state)
        y = np.asarray(y).astype(int)
        # Recompensas: acierto en alta, acierto en normal, falsa alarma, alta no detectada
        r_vp, r_vn, r_fp, r_fn = self.recompensas
        # Matriz de recompensas [acción][realidad]
        R = np.array([[r_vn, r_fn],           # acción 0 = "normal": recompensa si era normal / si era alta
                      [r_fp, r_vp]])          # acción 1 = "alta"  : recompensa si era normal / si era alta
        # Cuántas veces aparece cada producto
        conteo = X["Product ID"].value_counts()
        self.estados_ = {}
        for p in conteo[conteo >= self.min_obs].index:
            self.estados_[p] = len(self.estados_)
        for s in X["Sub-Category"].unique():
            self.estados_[s] = len(self.estados_)
        self.n_estados_ = len(self.estados_) + 1
        est = self._estado(X)

        # Tabla Q: valor esperado de cada acción (normal/alta) en cada estado, empieza en 0
        self.Q_ = np.zeros((self.n_estados_, 2))          # tabla Q: valor de cada acción en cada estado
        visitas = np.zeros((self.n_estados_, 2))
        # Aquí se guarda la recompensa promedio de cada episodio
        self.historial_ = []                               # recompensa promedio por episodio (curva de aprendizaje)
        n = len(y)
        # Cada episodio es una pasada completa por el entrenamiento
        for ep in range(self.episodios):
            # Epsilon baja con el tiempo: al inicio explora más, al final explota lo aprendido
            eps = self.epsilon_ini + (self.epsilon_fin - self.epsilon_ini) * ep / max(1, self.episodios - 1)
            # Recorremos los datos en orden aleatorio
            orden = rng.permutation(n)
            explora = rng.random(n) < eps
            azar = rng.integers(0, 2, n)
            total = 0.0
            # Cada fila es una decisión del agente
            for i, idx in enumerate(orden):
                s = est[idx]
                # Decisión: con probabilidad epsilon explora (acción al azar); si no, elige la de mayor Q
                a = azar[i] if explora[i] else int(self.Q_[s, 1] > self.Q_[s, 0])   # explorar o explotar
                # El entorno entrega la recompensa según acierto o error
                r = R[a, y[idx]]                                                     # recompensa recibida
                visitas[s, a] += 1
                # Actualización: acerca Q al promedio de recompensas recibidas
                self.Q_[s, a] += (r - self.Q_[s, a]) / visitas[s, a]                 # actualización de Q
                total += r
            self.historial_.append(total / n)
        self.classes_ = np.array([0, 1])
        return self

    # Convierte la ventaja de 'alta' sobre 'normal' en un puntaje entre 0 y 1
    def predict_proba(self, X):
        s = self._estado(X)
        dif = self.Q_[s, 1] - self.Q_[s, 0]                # ventaja de decir "alta" frente a "normal"
        p = 1 / (1 + np.exp(-dif))
        p = np.where(dif <= 0, np.minimum(p, 0.5 - 1e-9), p)   # si no hay ventaja, la política dice "normal"
        return np.c_[1 - p, p]

    def predict(self, X):
        s = self._estado(X)
        return (self.Q_[s, 1] > self.Q_[s, 0]).astype(int)
