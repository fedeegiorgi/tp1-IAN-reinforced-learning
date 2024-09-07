import argparse
from qlearning import AmbienteDiezMil, AgenteQLearning

GRID_SEARCH = False
RUN_AVG_TURN_TEST = False

def get_promedio_turnos(jugador, num_partidas, verbose=False) -> float:
    '''
    Juega num_partidas partidas con el jugador dado y devuelve el promedio de turnos necesarios.

    Args:
        jugador: Jugador a utilizar.
        num_partidas: Cantidad de partidas a jugar.
        verbose: Si se desea imprimir información adicional.

    Returns:
        float: Promedio de turnos necesarios para terminar una partida.
    '''
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
    '''
    Realiza una búsqueda de hiperparámetros para el agente Q-Learning.

    Args: 
        lr_range: Lista con los valores de learning rate a probar.
        gamma_range: Lista con los valores de gamma a probar.
        eps_range: Lista con los valores de epsilon a probar.
        episodios: Cantidad de episodios de entrenamiento.
        cant_partidas_promedio: Cantidad de partidas a jugar para obtener el promedio de turnos.
        verbose: Si se desea imprimir información adicional.

    Returns:
        float: Mejor learning rate.
        float: Mejor gamma.
        float: Mejor epsilon.
    '''
    ambiente = AmbienteDiezMil()
    mejor_promedio = math.inf

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
                    agente.guardar_politica('test_mejor.json')
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

    return best_lr, best_gamma, best_eps

def main(episodios, verbose):

    if GRID_SEARCH:
        lr_list = [0.05, 0.1, 0.2]
        gamma_list = [0.65, 0.7, 0.75, 0.8, 0.85]
        eps_list = [0.05, 0.1, 0.2]
        best_lr, best_gamma, best_eps = grid_search_hiperparametros(lr_list, gamma_list, eps_list, 1_000_000, 10000)

    if RUN_AVG_TURN_TEST:
        n_partidas = 100000
        jugador = JugadorEntrenado('QLearningAgent', 'best_training_policy.json')
        avg = get_promedio_turnos(jugador, n_partidas, verbose=True)
        print(f'Resultado obtenido con el agente que jugo {n_partidas} partidas: {avg}')

    ambiente = AmbienteDiezMil()
    agente = AgenteQLearning(ambiente, 0.05, 0.75, 0.2)
    agente.entrenar(episodios, verbose)
    agente.guardar_politica(f'policy_{episodios}.json')


if __name__ == '__main__':
    # Crear un analizador de argumentos
    parser = argparse.ArgumentParser(description="Entrenar un agente usando Q-learning en el ambiente de 'Diez Mil'.")

    # Agregar argumentos
    parser.add_argument('-e', '--episodios', type=int, default=10000, help='Número de episodios para entrenar al agente (default: 10000)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Activar modo verbose para ver más detalles durante el entrenamiento')

    # Parsear los argumentos
    args = parser.parse_args()

    # Llamar a la función principal con los argumentos proporcionados
    main(args.episodios, args.verbose)
