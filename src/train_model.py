"""
Módulo de Treinamento e Avaliação da Modelagem Estatística
Connecta Telecom - Previsão e Interpretação de Churn

Este script executa o pré-processamento dos dados, ajuste da Regressão Logística
via Statsmodels, cálculo das Razões de Chance (Odds Ratios) com intervalos de
confiança e métricas de desempenho estatísticas.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_and_preprocess_data(file_path: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carrega o dataset e aplica a codificação de variáveis dummy
    com definição explícita das categorias de referência.
    """
    if file_path is None or not os.path.exists(file_path):
        from data_generator import generate_churn_data
        df = generate_churn_data()
    else:
        df = pd.read_csv(file_path)

    # Criação de variáveis dummy com categorias de referência:
    # Tipo_Contrato: categoria base = 'Anual'
    # Servico_Internet: categoria base = 'DSL'
    df_encoded = pd.get_dummies(
        df,
        columns=['Tipo_Contrato', 'Servico_Internet'],
        drop_first=False,
        dtype=int
    )

    # Selecionamos as variáveis preditoras excluindo as de referência
    features = [
        'Fidelidade_Meses',
        'Fatura_Mensal',
        'Tipo_Contrato_Dois anos',
        'Tipo_Contrato_Mensal',
        'Servico_Internet_Fibra Óptica',
        'Servico_Internet_Não'
    ]

    X = df_encoded[features].copy()
    y = df_encoded['Churn'].copy()

    # Adicionamos a constante (intercepto beta_0)
    X = sm.add_constant(X)

    return X, y


def train_logistic_regression(X: pd.DataFrame, y: pd.Series):
    """
    Ajusta o modelo de Regressão Logística e exibe os resultados estatísticos.
    """
    modelo = sm.Logit(y, X)
    resultado = modelo.fit(disp=False)
    return resultado


def calculate_odds_ratios(resultado) -> pd.DataFrame:
    """
    Calcula a Razão de Chances (Odds Ratio), intervalos de confiança de 95% e p-valores.
    """
    params = resultado.params
    conf = resultado.conf_int()
    conf['Odds_Ratio'] = params
    conf.columns = ['IC_2.5%', 'IC_97.5%', 'Odds_Ratio']
    odds_df = np.exp(conf)
    odds_df['p_valor'] = resultado.pvalues
    return odds_df


def evaluate_model(resultado, X: pd.DataFrame, y: pd.Series):
    """
    Calcula métricas de classificação (Acurácia, Precisão, Sensibilidade/Recall, F1-Score)
    para o limiar padrão de 0.50.
    """
    y_prob = resultado.predict(X)
    y_pred = (y_prob >= 0.50).astype(int)

    tp = np.sum((y_pred == 1) & (y == 1))
    tn = np.sum((y_pred == 0) & (y == 0))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))

    accuracy = (tp + tn) / len(y)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 60)
    print("AVALIAÇÃO DE DESEMPENHO DO MODELO (Threshold = 0.50)")
    print("=" * 60)
    print(f"Verdadeiros Negativos (TN): {tn:4d}  | Falsos Positivos   (FP): {fp:4d}")
    print(f"Falsos Negativos       (FN): {fn:4d}  | Verdadeiros Positivos (TP): {tp:4d}")
    print("-" * 60)
    print(f"Acurácia Geral : {accuracy * 100:.2f}%")
    print(f"Precisão       : {precision * 100:.2f}%")
    print(f"Recall (Sensib): {recall * 100:.2f}%")
    print(f"F1-Score       : {f1 * 100:.2f}%")
    print(f"Pseudo R² (McF): {resultado.prsquared:.4f}")
    print(f"AIC            : {resultado.aic:.2f}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'connecta_telecom_churn.csv')

    print("Carregando e preparando dados...")
    X, y = load_and_preprocess_data(csv_path)

    print("Ajustando Regressão Logística via Statsmodels...")
    resultado = train_logistic_regression(X, y)

    print("\n" + "=" * 60)
    print("RESUMO ESTATÍSTICO DO MODELO")
    print("=" * 60)
    print(resultado.summary())

    print("\n" + "=" * 60)
    print("TABELA DE RAZÃO DE CHANCES (ODDS RATIOS) & INTERVALOS DE CONFIANÇA")
    print("=" * 60)
    odds_df = calculate_odds_ratios(resultado)
    print(odds_df.round(4).to_string())

    evaluate_model(resultado, X, y)


if __name__ == '__main__':
    main()
