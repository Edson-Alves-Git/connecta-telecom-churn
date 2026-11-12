"""
Módulo de Geração de Dados Sintéticos de Churn
Connecta Telecom

Este módulo gera uma base de dados realista de 2.000 clientes simulando
o comportamento de cancelamento (churn) baseado em regras de negócio estatísticas.
"""

import numpy as np
import pandas as pd


def generate_churn_data(num_clientes: int = 2000, random_seed: int = 42) -> pd.DataFrame:
    """
    Gera um DataFrame com dados de clientes de telecomunicações.

    Parâmetros:
    -----------
    num_clientes : int, padrão 2000
        Quantidade de registros de clientes a serem gerados.
    random_seed : int, padrão 42
        Semente para garantia de reprodutibilidade dos dados.

    Retorno:
    --------
    pd.DataFrame
        DataFrame contendo as colunas:
        - ID_Cliente: Identificador único do cliente
        - Fidelidade_Meses: Tempo de permanência do cliente na operadora (1 a 72 meses)
        - Tipo_Contrato: Modalidade de contrato ('Mensal', 'Anual', 'Dois anos')
        - Servico_Internet: Tecnologia de internet ('Fibra Óptica', 'DSL', 'Não')
        - Fatura_Mensal: Valor da fatura em R$ (20.00 a 120.00)
        - Churn: Indicador de cancelamento (0 = Não, 1 = Sim)
    """
    np.random.seed(random_seed)

    fidelidade_meses = np.random.randint(1, 73, size=num_clientes)
    tipo_contrato_opts = ['Mensal', 'Anual', 'Dois anos']
    contrato_probs = [0.60, 0.25, 0.15]
    tipo_contrato = np.random.choice(tipo_contrato_opts, size=num_clientes, p=contrato_probs)

    servico_internet_opts = ['Fibra Óptica', 'DSL', 'Não']
    internet_probs = [0.55, 0.35, 0.10]
    servico_internet = np.random.choice(servico_internet_opts, size=num_clientes, p=internet_probs)

    fatura_base = {
        'Mensal': np.random.normal(60, 20),
        'Anual': np.random.normal(70, 25),
        'Dois anos': np.random.normal(80, 25)
    }

    fatura_mensal = [
        fatura_base[c] + fidelidade_meses[i] * 0.2 + np.random.normal(0, 5)
        for i, c in enumerate(tipo_contrato)
    ]
    fatura_mensal = np.clip(fatura_mensal, 20, 120)

    # Função geradora do log-odds de Churn
    prob_churn_log = -2.5
    prob_churn_log += -0.05 * fidelidade_meses
    prob_churn_log += [3.0 if c == 'Mensal' else -1.5 if c == 'Anual' else -2.5 for c in tipo_contrato]
    prob_churn_log += [0.8 if s == 'Fibra Óptica' else -0.5 for s in servico_internet]
    prob_churn_log += 0.03 * fatura_mensal

    # Transformação logística sigmoide para obter probabilidades reais
    prob_churn = 1 / (1 + np.exp(-prob_churn_log))
    churn = np.random.binomial(1, prob_churn)

    df = pd.DataFrame({
        'ID_Cliente': range(1, num_clientes + 1),
        'Fidelidade_Meses': fidelidade_meses,
        'Tipo_Contrato': tipo_contrato,
        'Servico_Internet': servico_internet,
        'Fatura_Mensal': np.round(fatura_mensal, 2),
        'Churn': churn
    })

    return df


if __name__ == '__main__':
    data = generate_churn_data()
    print("Amostra dos Dados:")
    print(data.head())
    print("\nTaxa Geral de Churn:", f"{(data['Churn'].mean() * 100):.2f}%")
