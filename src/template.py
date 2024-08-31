import json
import random
import numpy as np
from tqdm import tqdm
from jugador import Jugador
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR

'''
self.state_matrix: dict[EstadoDiezMil, tuple[float, float]]
{
f'cant_dados: {self.dados} | puntos_turno: {self.pts_turno} | puntos_totales: {self.pts_total}': (Q-VALUE PLANTARSE, Q-VALUE SEGUIR)
(STATE 2) : (Q-VALUE PLANTARSE, Q-VALUE SEGUIR)
...
(STATE N) : (Q-VALUE PLANTARSE, Q-VALUE SEGUIR)
}
'''

class AmbienteDiezMil:
    def __init__(self):
        """
        Definir las variables de instancia de un ambiente.
        ¿Qué es propio de un ambiente de 10.000?
        """

        self.turno_actual = 1
        self.estado_actual = EstadoDiezMil(6, 0)
        self.puntos_totales = 0
        self.min_max_cara_dado = [1, 6]
        self.acciones_posibles = [JUGADA_PLANTARSE, JUGADA_TIRAR]

    def tirada(self, cant_dados):
        """
        Devuelve los resultados de una tirada al pedirsela al ambiente.
        """

        min_cara, max_cara = self.min_max_cara_dado
        return [random.randint(min_cara, max_cara) for _ in range(cant_dados)]

    def reset(self):
        """
        Reinicia el ambiente para volver a realizar un episodio.
        """

        self.turno_actual = 1
        self.estado_actual = EstadoDiezMil(6, 0)
        self.puntos_totales = 0

    def step(self, accion):
        """
        Dada una acción devuelve una recompensa.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Podría ser útil devolver si terminó o no el turno.

        Args:
            accion: Acción elegida por un agente.

        Returns:
            tuple[int, bool]: Una recompensa y un flag que indica si terminó el episodio.
        """

        if accion == JUGADA_PLANTARSE:
            self.puntos_totales += self.estado_actual.puntos_turno
            if self.puntos_totales >= 10000:
                cant_turnos = self.turno_actual
                self.reset()
                return (-cant_turnos, True)
            else:
                self.estado_actual.fin_turno()
                self.turno_actual += 1
                return (0, False)
        else:
            tirada = self.tirada(self.estado_actual.dados)
            puntos_tirada, dados_restantes = puntaje_y_no_usados(tirada)
            self.estado_actual.dados = len(dados_restantes)
            if puntos_tirada == 0:
                self.estado_actual.fin_turno()
                self.turno_actual += 1
                return (0, False)
            else:
                self.estado_actual.puntos_turno += puntos_tirada
                return (0, False)


class EstadoDiezMil:
    def __init__(self, dados, puntos_turno):
        """
        Definir qué hace a un estado de diez mil.
        Recordar que la complejidad del estado repercute en la complejidad de la tabla del agente de q-learning.
        """

        self.dados = dados
        self.puntos_turno = puntos_turno

    def fin_turno(self):
        """
        Modifica el estado al terminar el turno.
        """

        self.puntos_turno = 0
        self.dados = 6

    def __str__(self):
        """
        Representación en texto de EstadoDiezMil.
        Ayuda a tener una versión legible del objeto.

        Returns:
            str: Representación en texto de EstadoDiezMil.
        """

        return f'cant_dados: {self.dados} | puntos_turno: {self.puntos_turno}'

class AgenteQLearning:
    def __init__(
        self,
        ambiente: AmbienteDiezMil,
        alpha: float,
        gamma: float,
        epsilon: float,
        *args,
        **kwargs
    ):
        """
        Definir las variables internas de un Agente que implementa el algoritmo de Q-Learning.

        Args:
            ambiente (AmbienteDiezMil): Ambiente con el que interactuará el agente.
            alpha (float): Tasa de aprendizaje.
            gamma (float): Factor de descuento.
            epsilon (float): Probabilidad de explorar.
        """

        self.qlearning_tabla: dict[str, list[float]] = {}
        self.ambiente = ambiente
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def elegir_accion(self, eps_greedy=True):
        """
        Selecciona una acción de acuerdo a una política ε-greedy.
        """
        qlearn_key = str(self.ambiente.estado_actual)
        if qlearn_key not in self.qlearning_tabla.keys():
            self.qlearning_tabla[qlearn_key] = [-25, -25]

        # Empate entre q-values
        if self.qlearning_tabla[qlearn_key][0] == self.qlearning_tabla[qlearn_key][1]:
            return random.randint(0, 1)

        # Veo cual es la decision a tomar en caso de que salga explorar (p = epsilon)
        decision_explorar = np.argmin(self.qlearning_tabla[qlearn_key])

        if eps_greedy and random.uniform(0, 1) < self.epsilon:
            # Si sale explorar, exploro, si no, tomo la otra decision (notar que son solo 2 decisiones posibles)
            return decision_explorar

        return 1 - decision_explorar

    def actualizar_tabla(self, key, recompensa, accion_elegida):
        estimacion_error = self.qlearning_tabla[key][accion_elegida]
        qlearn_siguiente_key = str(self.ambiente.estado_actual)
        if qlearn_siguiente_key not in self.qlearning_tabla.keys():
            self.qlearning_tabla[qlearn_siguiente_key] = [-25, -25]
        max_q = self.elegir_accion(eps_greedy=False)
        self.qlearning_tabla[key][accion_elegida] += self.alpha * (recompensa + self.gamma * max_q - estimacion_error)

    def entrenar(self, episodios: int, verbose: bool = False) -> None:
        """
        Dada una cantidad de episodios, se repite el ciclo del algoritmo de Q-learning.
        Recomendación: usar tqdm para observar el progreso en los episodios.

        Args:
            episodios (int): Cantidad de episodios a iterar.
            verbose (bool, optional): Flag para hacer visible qué ocurre en cada paso. Defaults to False.
        """

        for _ in tqdm(range(episodios)):
            termino_episodio = False
            while not termino_episodio:
                accion_elegida = self.elegir_accion()
                qlearn_key = str(self.ambiente.estado_actual)
                if qlearn_key not in self.qlearning_tabla.keys():
                    self.qlearning_tabla[qlearn_key] = [-25, -25]
                recompensa, termino_episodio = self.ambiente.step(accion_elegida)
                self.actualizar_tabla(qlearn_key, recompensa, accion_elegida)

    def guardar_politica(self, filename: str):
        """
        Almacena la política del agente en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo a generar.
        """

        with open(filename, 'w') as jsonfile:
            json.dump(self.qlearning_tabla, jsonfile, indent=4)

class JugadorEntrenado(Jugador):
    def __init__(self, nombre: str, filename_politica: str):
        self.nombre = nombre
        self.politica = self._leer_politica(filename_politica)

    def _leer_politica(self, filename: str, SEP: str = ','):
        """
        Carga una politica entrenada con un agente de RL, que está guardada
        en el archivo filename en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo que contiene a una política almacenada. 
        """

        with open(filename, 'r') as jsonfile:
            politica = json.load(jsonfile)

        return politica

    def jugar(
        self,
        puntaje_turno: int,
        dados: list[int],
    ) -> tuple[int, list[int]]:
        """
        Devuelve una jugada y los dados a tirar.

        Args:
            puntaje_total (int): Puntaje total del jugador en la partida.
            puntaje_turno (int): Puntaje en el turno del jugador
            dados (list[int]): Tirada del turno.

        Returns:
            tuple[int,list[int]]: Una jugada y la lista de dados a tirar.
        """
        puntaje, no_usados = puntaje_y_no_usados(dados)

        estado = f'cant_dados: {len(no_usados)} | puntos_turno: {puntaje_turno + puntaje}'
        jugada = self.politica[estado]

        if jugada == JUGADA_PLANTARSE:
            return (JUGADA_PLANTARSE, [])
        elif jugada == JUGADA_TIRAR:
            return (JUGADA_TIRAR, no_usados)


'''
cant_dados [0, 1, 2, 3, 4, 5, 6] = 7
puntos_turno [0, 50, 100, 150, ... 10000] = 10000 / 50 = 200
puntos_totales [0, 50, 100, ... 20000] = 20000 / 50 = 400
'''
