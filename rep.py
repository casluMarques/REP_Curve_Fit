import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 1. Funções de ajuste
def linear_func(x, a, b):
    return a * x + b

def log_func(x, a, b):
    return a * np.log(x) + b

def quadratic_func(x, a, b, c):
    return a * (x**2) + b * x + c

def get_user_input():
    """Coleta os pares (Número, Tempo) para treino e um opcional para teste."""
    numbers = []
    times = []
    
    # --- NOVO: Solicitar a quantidade de pontos desejada ao usuário ---
    while True:
        try:
            n_input = input("Quantos pontos de calibração você deseja inserir? (Mínimo 2): ").strip()
            max_measurements = int(n_input)
            if max_measurements >= 2:
                break
            print("Você precisa de pelo menos 2 pontos para gerar as curvas de regressão.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número inteiro.")
            
    print("\n--- Entrada de Dados de Calibração ---")
    print("OBS: Insira o tempo em SEGUNDOS (o script converterá para minutos).")
    print(f"Insira até {max_measurements} medições (Número e Tempo).")

    for i in range(max_measurements):
        try:
            num_input = input(f"Medição {i+1} - Insira o NÚMERO (ou 's' para parar): ").strip()
            if num_input.lower() == 's' and i >= 2:
                break
            elif num_input.lower() == 's' and i < 2:
                print("Mínimo de 2 medições necessárias.")
                continue

            time_input = input(f"Medição {i+1} - Insira o TEMPO (segundos): ").strip()

            number = float(num_input)
            time = float(time_input)

            if number <= 0:
                print("O Número deve ser positivo.")
                continue

            numbers.append(number)
            times.append(time)

        except ValueError:
            print("Entrada inválida.")
            continue
        except EOFError:
            break
            
    if len(numbers) < 2:
        return None, None, None, None, None

    while True:
        try:
            x_predict = float(input("\nInsira o NÚMERO alvo para a PREVISÃO FUTURA (ex: 480): "))
            if x_predict > 0: break
            print("Deve ser positivo.")
        except ValueError:
            print("Inválido.")

    # --- NOVO: Ponto de Teste/Validação ---
    x_test = None
    y_test = None
    print("\n--- Ponto de Teste / Validação (Opcional) ---")
    print("Insira os dados do ponto que você JÁ CONHECE para comparar com as curvas.")
    try:
        test_num_input = input("Número conhecido (Ex: 100) ou [Enter] para pular: ").strip()
        if test_num_input != "":
            x_test = float(test_num_input)
            test_time_input = input(f"Insira o TEMPO real para as {x_test} marcações (segundos): ").strip()
            y_test = float(test_time_input)
    except ValueError:
        print("Entrada inválida. Plotagem do ponto de teste ignorada.")
        x_test, y_test = None, None

    return np.array(numbers), np.array(times), x_predict, x_test, y_test

# 2. Função principal
def run_curve_fitting():
    # Recebe agora também X_test e Y_test
    X, Y, X_predict, X_test, Y_test = get_user_input()
    
    if X is None:
        return

    # Converte para minutos
    Y = Y / 60 
    print(f"\n[Info] Dados convertidos para MINUTOS.")

    # Converte o ponto de teste para minutos, se ele existir
    if y_test_exists := (X_test is not None and Y_test is not None):
        Y_test = Y_test / 60

    results = {}
    
    # Lista de modelos
    models = [
        ("Linear", linear_func),
        ("Logarítmica", log_func),
        ("Quadrática", quadratic_func)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, Y, color='black', label='Dados Originais (Treino)', zorder=5)

    # Plotar o ponto de teste no gráfico (se o usuário informou)
    if y_test_exists:
        ax.scatter(X_test, Y_test, color='magenta', marker='*', s=200, edgecolor='black', 
                   label=f'Ponto Real ({X_test} marc. = {Y_test:.2f} min)', zorder=6)

    # Determinar até onde a linha do gráfico deve ir no eixo X
    max_x_plot = X_predict
    if y_test_exists and X_test > X_predict:
        max_x_plot = X_test

    for name, func in models:
        try:
            if name == "Quadrática":
                popt, pcov = curve_fit(func, X, Y, bounds=(0, np.inf))
            else:
                popt, pcov = curve_fit(func, X, Y)
            
            # Previsão
            Y_predict = func(X_predict, *popt)
            
            if Y_predict < 0:
                Y_predict = 0.0

            # Equações
            if name == "Linear":
                equation = f"y = {popt[0]:.6f}x + {popt[1]:.6f}"
            elif name == "Logarítmica":
                equation = f"y = {popt[0]:.6f}ln(x) + {popt[1]:.6f}"
            elif name == "Quadrática":
                equation = f"y = {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f}"

            # Se houver ponto de teste, calcular também a previsão neste ponto para avaliar o erro
            test_error_str = "N/A"
            if y_test_exists:
                pred_at_test = max(func(X_test, *popt), 0)
                erro = abs(pred_at_test - Y_test)
                test_error_str = f"{erro:.2f} min"

            results[name] = {
                "Predição (Min)": Y_predict,
                "Equação": equation,
                "Erro Abs. no Ponto Teste": test_error_str
            }
            
            # Ajustando o domínio da linha traçada no gráfico
            X_fit = np.linspace(X.min(), max_x_plot * 1.05, 100)
            Y_fit = func(X_fit, *popt)
            Y_fit = np.maximum(Y_fit, 0)
            
            ax.plot(X_fit, Y_fit, label=f'{name}')
            ax.scatter(X_predict, Y_predict, marker='X', s=100, label=f'{name}: {Y_predict:.2f}')

        except Exception as e:
            results[name] = {"Predição (Min)": "Erro", "Equação": str(e), "Erro Abs. no Ponto Teste": "Erro"}
            print(f"Erro em {name}: {e}")

    # Ajustes finais do gráfico
    ax.set_xlim(0, max_x_plot * 1.1)
    
    # Limite Y dinâmico para garantir que todos os pontos e curvas apareçam
    vals = [r["Predição (Min)"] for r in results.values() if isinstance(r["Predição (Min)"], (int, float))]
    all_y = vals + [Y.max()]
    if y_test_exists:
        all_y.append(Y_test)
        
    y_max = max(all_y) * 1.2 if all_y else Y.max() * 1.5
    
    ax.set_ylim(0, y_max)
    
    ax.set_title(f'Previsão Não-Negativa (Alvo X = {X_predict})')
    ax.set_xlabel('Número (Marcações)')
    ax.set_ylabel('Tempo (Minutos)')
    ax.legend()
    plt.grid(True, alpha=0.3)
    
    output_filename = 'curve_fit_non_negative.png'
    plt.savefig(output_filename)
    plt.close(fig)

    # Tabela
    df = pd.DataFrame.from_dict(results, orient='index')
    print("\n" + "="*70)
    print(f"Resultados de Predição para X = {X_predict}")
    print("="*70)
    try:
        print(df.to_markdown())
    except ImportError:
        print(df.to_string())
    print(f"\nGráfico atualizado gerado: {output_filename}")

if __name__ == "__main__":
    run_curve_fitting()