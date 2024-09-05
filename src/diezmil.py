import os
import math
import copy
from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt
from utils import puntaje_y_no_usados, separar, JUGADA_PLANTARSE, JUGADA_TIRAR
from jugador import Jugador, JugadorAleatorio, JugadorSiempreSePlanta
from template import AmbienteDiezMil, EstadoDiezMil, AgenteQLearning, JugadorEntrenado

TRAIN = True
RUN_AVG_TURN_TEST = False
GRAFICO = True

class JuegoDiezMil:
    def __init__(self, jugador: Jugador):
        self.jugador: Jugador = jugador

    def jugar(self, verbose: bool = False, tope_turnos: int = 1000) -> tuple[int, int]:
        ''' Juega un juego de 10mil para un jugador, hasta terminar o hasta
            llegar a tope_turnos turnos. Devuelve la cantidad de turnos que
            necesitó y el puntaje final.
        '''
        turno: int = 0
        puntaje_total: int = 0
        while puntaje_total < 10000 and turno < tope_turnos:
            # Nuevo turno
            turno += 1
            puntaje_turno: int = 0
            msg: str = 'turno ' + str(turno) + ':'

            # Un turno siempre empieza tirando los 6 dados.
            jugada: int = JUGADA_TIRAR
            dados_a_tirar: list[int] = [1, 2, 3, 4, 5, 6]
            fin_de_turno: bool = False

            while not fin_de_turno:
                # Tira los dados que correspondan y calcula su puntaje.
                dados: list[int] = [randint(1, 6) for _ in range(len(dados_a_tirar))]
                (puntaje_tirada, _) = puntaje_y_no_usados(dados)
                msg += ' ' + ''.join(map(str, dados)) + ' '

                if puntaje_tirada == 0:
                    # Mala suerte, no suma nada. Pierde el turno.
                    fin_de_turno = True
                    puntaje_turno = 0

                else:
                    # Bien, suma puntos. Preguntamos al jugador qué quiere hacer.
                    jugada, dados_a_tirar = self.jugador.jugar(puntaje_turno, dados)

                    if jugada == JUGADA_PLANTARSE:
                        msg += 'P'
                        fin_de_turno = True
                        puntaje_turno += puntaje_tirada

                    elif jugada == JUGADA_TIRAR:
                        dados_a_separar = separar(dados, dados_a_tirar)
                        assert len(dados_a_separar) + len(dados_a_tirar) == len(dados)
                        puntaje_tirada, dados_no_usados = puntaje_y_no_usados(dados_a_separar)
                        assert puntaje_tirada > 0 and len(dados_no_usados) == 0
                        puntaje_turno += puntaje_tirada
                        # Cuando usó todos los dados, vuelve a tirar todo.
                        if len(dados_a_tirar) == 0:
                            dados_a_tirar = [1, 2, 3, 4, 5, 6]
                        msg += 'T(' + ''.join(map(str, dados_a_tirar)) + ') '

            puntaje_total += puntaje_turno
            msg += ' --> ' + str(puntaje_turno) + ' puntos. TOTAL: ' + str(puntaje_total)
            if verbose:
                print(msg)
        return (turno, puntaje_total)

def get_promedio_turnos(jugador, num_partidas, verbose=False):
    avg = 0
    if verbose:
        for _ in tqdm(range(num_partidas)):
            juego = JuegoDiezMil(jugador)
            (cantidad_turnos, puntaje_final) = juego.jugar(verbose=False)
            avg += cantidad_turnos
    else:
        for _ in range(num_partidas):
            juego = JuegoDiezMil(jugador)
            (cantidad_turnos, puntaje_final) = juego.jugar(verbose=False)
            avg += cantidad_turnos
    return avg / num_partidas

def grid_search_hiperparametros(lr_range, gamma_range, eps_range, episodios, cant_partidas_promedio, verbose=True):
    ambiente = AmbienteDiezMil()
    mejor_promedio = math.inf
    mejor_agente = None

    for lr in lr_range:
        for gamma in gamma_range:
            for eps in eps_range:
                if verbose:
                    print(f'Probando con: LR = {lr:.2f} | Gamma: {gamma:.2f} | Epsilon: {eps:.2f}')
                agente = AgenteQLearning(ambiente, lr, gamma, eps)
                agente.entrenar(episodios)
                agente.guardar_politica('test_policy.json')
                jugador = JugadorEntrenado('TestAgent', 'test_policy.json')
                turnos_promedio = get_promedio_turnos(jugador, cant_partidas_promedio)
                if turnos_promedio < mejor_promedio:
                    mejor_promedio = turnos_promedio
                    agente.guardar_politica('best_training_policy.json')
                    best_progreso = agente.progreso
                    best_lr, best_gamma, best_eps = lr, gamma, eps
                    if verbose:
                        print(f'Nuevo mejor promedio obtenido: {turnos_promedio}. LR: {lr:.2f} | Gamma: {gamma:.2f} | Epsilon: {eps:.2f}')
                else:
                    if verbose:
                        print(f'Promedio obtenido: {turnos_promedio} [LR: {lr:.2f} | Gamma: {gamma:.2f} | Epsilon: {eps:.2f}]')
    
    # Si el archivo se corre por fuera de la carpeta y el archivo se genera afuera, no borrarlo pero no tirar error.
    try:
        os.remove('test_policy.json')
    except Exception:
        pass

    if verbose:
        print(f'Mejores hiperparametros obtenidos: LR: {best_lr} | Gamma: {best_gamma} | Epsilon: {best_eps}')

    return best_lr, best_gamma, best_eps, best_progreso

def main():
    if TRAIN:
        lr_list = [0.05, 0.1, 0.2]
        gamma_list = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1]
        eps_list = [0.05, 0.1, 0.2]
        best_lr, best_gamma, best_eps, best_progreso = grid_search_hiperparametros(lr_list, gamma_list, eps_list, 2, 2)

        if GRAFICO:
            plt.plot(best_progreso)
            plt.title('Progreso del mejor agente entrenado')
            plt.xlabel('Episodio')
            plt.ylabel('Turnos al terminar')
            plt.savefig('progreso_agente_entrenado.png')

    if RUN_AVG_TURN_TEST:
        n_partidas = 100000
        jugador = JugadorEntrenado('QLearningAgent', 'best_training_policy.json')
        avg = get_promedio_turnos(jugador, n_partidas, verbose=True)
        print(f'Resultado obtenido con el agente que jugó {n_partidas} partidas: {avg}')

if __name__ == '__main__':
    main()
