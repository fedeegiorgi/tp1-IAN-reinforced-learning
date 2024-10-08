import json
import random
import numpy as np
from tqdm import tqdm
from jugador import Jugador
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR

class AmbienteDiezMil:
    def __init__(self):
        '''
        Definir las variables internas de un ambiente de Diez Mil.
        '''

        self.turno_actual = 1
        self.estado_actual = EstadoDiezMil(6, 0)
        self.puntos_totales = 0
        self.min_max_cara_dado = [1, 6]
        self.acciones_posibles = [JUGADA_PLANTARSE, JUGADA_TIRAR]

    def tirada(self, cant_dados) -> list[int]:
        '''
        Devuelve los resultados de una tirada al pedirsela al ambiente.

        Args:
            cant_dados (int): Cantidad de dados a tirar.
        
        Returns:
            list[int]: Resultados de la tirada.
        '''

        min_cara, max_cara = self.min_max_cara_dado
        return [random.randint(min_cara, max_cara) for _ in range(cant_dados)]

    def reset(self):
        '''
        Reinicia el ambiente para volver a realizar un episodio.
        '''

        self.turno_actual = 1
        self.estado_actual = EstadoDiezMil(6, 0)
        self.puntos_totales = 0

    def step(self, accion) -> tuple[int, bool]:
        '''
        Dada una acción devuelve una recompensa.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Podría ser útil devolver si terminó o no el turno.

        Args:
            accion: Acción elegida por un agente.

        Returns:
            tuple[int, bool]: Una recompensa y un flag que indica si terminó el episodio.
        '''
        assert accion in self.acciones_posibles
        recompensa = self.estado_actual.puntos_turno
        partida_terminada = False
        turnos_al_terminar = 0

        if accion == JUGADA_PLANTARSE:
            self.puntos_totales += self.estado_actual.puntos_turno

            if self.puntos_totales >= 10000:
                partida_terminada = True
                recompensa = 10000 / self.turno_actual
                self.reset()
            else:
                self.estado_actual.fin_turno()
                self.turno_actual += 1
        else:
            tirada = self.tirada(self.estado_actual.dados)
            puntos_tirada, dados_restantes = puntaje_y_no_usados(tirada)

            if puntos_tirada == 0:
                if self.estado_actual.dados == 0:
                    recompensa = -self.estado_actual.puntos_turno / 6
                else:
                    recompensa = -self.estado_actual.puntos_turno / (self.estado_actual.dados)
                self.estado_actual.fin_turno()
                self.turno_actual += 1
            else:
                self.estado_actual.dados = len(dados_restantes)
                self.estado_actual.puntos_turno += puntos_tirada

        return recompensa, partida_terminada


class EstadoDiezMil:
    def __init__(self, dados, puntos_turno):
        '''
        Define el estado de un juego de Diez Mil.

        Args:
            int: Cantidad de dados disponibles.
            int: Puntos acumulados en el turno.
        '''

        self.dados = dados
        self.puntos_turno = puntos_turno

    def fin_turno(self):
        '''
        Modifica el estado al terminar el turno.
        '''

        self.puntos_turno = 0
        self.dados = 6

    def __str__(self):
        '''
        Representación en texto de EstadoDiezMil.
        Ayuda a tener una versión legible del objeto.

        Returns:
            str: Representación en texto de EstadoDiezMil.
        '''

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
        '''
        Definir las variables internas de un Agente que implementa el algoritmo de Q-Learning.

        Args:
            ambiente (AmbienteDiezMil): Ambiente con el que interactuará el agente.
            alpha (float): Tasa de aprendizaje.
            gamma (float): Factor de descuento.
            epsilon (float): Probabilidad de explorar.
        '''

        self.qlearning_tabla: dict[str, list[float]] = {}
        self.ambiente = ambiente
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def elegir_accion(self, eps_greedy=True):
        '''
        Selecciona una acción de acuerdo a una política ε-greedy.
        '''
        qlearn_key = str(self.ambiente.estado_actual)

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
        q_actual = self.qlearning_tabla[key][accion_elegida]
        qlearn_siguiente_key = str(self.ambiente.estado_actual)
        max_a = self.elegir_accion(eps_greedy=False)
        max_q = self.qlearning_tabla[qlearn_siguiente_key][max_a]
        self.qlearning_tabla[key][accion_elegida] += self.alpha * (recompensa + self.gamma * max_q - q_actual)

    def entrenar(self, episodios: int, verbose: bool = False) -> None:
        '''
        Dada una cantidad de episodios, se repite el ciclo del algoritmo de Q-learning.
        Recomendación: usar tqdm para observar el progreso en los episodios.

        Args:
            episodios (int): Cantidad de episodios a iterar.
            verbose (bool, optional): Flag para hacer visible qué ocurre en cada paso. Defaults to False.
        '''
        for N in range(7):
            for Y in range(0, 20001, 50):

                key = f'cant_dados: {N} | puntos_turno: {Y}'

                self.qlearning_tabla[key] = [0, 0]

        if verbose:
            rango_episodios = tqdm(range(episodios))
        else:
            rango_episodios = range(episodios)

        for _ in rango_episodios:
            termino_episodio = False
            while not termino_episodio:
                accion_elegida = self.elegir_accion()
                qlearn_key = str(self.ambiente.estado_actual)
                recompensa, termino_episodio = self.ambiente.step(accion_elegida)
                self.actualizar_tabla(qlearn_key, recompensa, accion_elegida)

    def guardar_politica(self, filename: str):
        '''
        Almacena la política del agente en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo a generar.
        '''

        with open(filename, 'w') as jsonfile:
            json.dump(self.qlearning_tabla, jsonfile, indent=4)

class JugadorEntrenado(Jugador):
    def __init__(self, nombre: str, filename_politica: str):
        self.nombre = nombre
        self.politica = self._leer_politica(filename_politica)

    def _leer_politica(self, filename: str, SEP: str = ','):
        '''
        Carga una politica entrenada con un agente de RL, que está guardada
        en el archivo filename en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo que contiene a una política almacenada. 
        '''

        with open(filename, 'r') as jsonfile:
            politica = json.load(jsonfile)

        return politica

    def jugar(
        self,
        puntaje_turno: int,
        dados: list[int],
    ) -> tuple[int, list[int]]:
        '''
        Devuelve una jugada y los dados a tirar.

        Args:
            puntaje_total (int): Puntaje total del jugador en la partida.
            puntaje_turno (int): Puntaje en el turno del jugador
            dados (list[int]): Tirada del turno.

        Returns:
            tuple[int,list[int]]: Una jugada y la lista de dados a tirar.
        '''
        nuevos_puntos, no_usados = puntaje_y_no_usados(dados)

        estado = f'cant_dados: {len(no_usados)} | puntos_turno: {puntaje_turno + nuevos_puntos}'
        jugada = self.politica[estado]
        jugada = np.argmax(jugada)

        if jugada == 0:
            return (JUGADA_PLANTARSE, [])
        elif jugada == 1:
            return (JUGADA_TIRAR, no_usados)
