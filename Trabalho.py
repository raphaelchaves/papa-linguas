from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import random
import math
import statistics
import csv
import matplotlib.pyplot as plt
import streamlit as st
import io
import numpy as np


# --- Estilo CSS e Barra lateral personalizados ---
def apply_custom_style():
    page_bg_img = """
    <style>
    .stApp {
        background-color: #f7f7f7;
        color: black;
    }
    div.stButton > button {
        color: black;
        background-color: #d1d1d1;
        padding: 10px 20px;
        border-radius: 5px;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] * {
        color: black;
    }
    </style>
    """
    st.markdown(sidebar_style, unsafe_allow_html=True)

    st.sidebar.image(
        "https://github.com/raphaelchaves/papa-linguas/raw/main/ICMC.png",
        width=150
    )
    st.sidebar.title("Sobre o App")
    st.sidebar.info(
        """
        Este APP foi desenvolvido para a disciplina de **MAI5001 - Introdução à Ciência de Computação**.
        """
    )
    st.sidebar.subheader("Componentes do Grupo:")
    st.sidebar.write("- Fábio Luíz Souza Alves")
    st.sidebar.write("- Júnior Fernandes Marques")
    st.sidebar.write("- Raphael Franco Chaves")

    st.sidebar.subheader("Docentes:")
    st.sidebar.write("- Cláudio Fabiano Motta Toledo")

    st.title("Grupo AlgoritmoExperts")
    st.write("Os Segredos da Análise Estatística")


# --- Classe DataGenerator ---
class DataGenerator:
    """Gera uma lista de floats aleatórios para ser usada pelo kit de estatística."""
    def __init__(self, n: int = 50, low: float = 0.0, high: float = 100.0,
                 mode: str = "uniform", seed: Optional[int] = 42):
        assert n > 0, "n deve ser positivo"
        assert low < high, "low < high é obrigatório"
        assert mode in {"uniform", "normal"}, "mode deve ser 'uniform' ou 'normal'"
        self.n, self.low, self.high, self.mode, self.seed = n, low, high, mode, seed

    def generate(self) -> List[float]:
        if self.seed is not None:
            random.seed(self.seed)
        if self.mode == "uniform":
            return [random.uniform(self.low, self.high) for _ in range(self.n)]
        mu = 0.5 * (self.low + self.high)
        sigma = max((self.high - self.low) / 6.0, 1e-9)
        data = [random.gauss(mu, sigma) for _ in range(self.n)]
        return [min(self.high, max(self.low, x)) for x in data]


# --- Classe CSVSource ---
class CSVSource:
    def __init__(self, filename: str, column: Union[int, str]):
        self.filename = filename
        self.column = column

    def read(self) -> List[float]:
        data = []
        with open(self.filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            if isinstance(self.column, str):
                col_index = headers.index(self.column)
            else:
                col_index = self.column
            for row in reader:
                try:
                    value = float(row[col_index])
                    data.append(value)
                except (ValueError, IndexError):
                    continue
        return data


# --- Classe estatística abstrata e subclasses ---
class Statistic(ABC):
    def __init__(self):
        self.value = None

    @abstractmethod
    def compute(self, data: List[float]):
        pass


class Mean(Statistic):
    def compute(self, data: List[float]):
        self.value = statistics.mean(data)
        return self.value


class Median(Statistic):
    def compute(self, data: List[float]):
        self.value = statistics.median(data)
        return self.value


class StdDev(Statistic):
    def compute(self, data: List[float]):
        self.value = statistics.stdev(data)
        return self.value


class MinValue(Statistic):
    def compute(self, data: List[float]):
        self.value = min(data)
        return self.value


class MaxValue(Statistic):
    def compute(self, data: List[float]):
        self.value = max(data)
        return self.value


class Variance(Statistic):
    def compute(self, data: List[float]):
        self.value = statistics.variance(data)
        return self.value


class Percentile25(Statistic):
    def compute(self, data: List[float]):
        self.value = statistics.quantiles(data, n=4)[0]
        return self.value


class Percentile75(Statistic):
    def compute(self, data: List[float]):
        self.value = statistics.quantiles(data, n=4)[2]
        return self.value


class Range(Statistic):
    def compute(self, data: List[float]):
        self.value = max(data) - min(data)
        return self.value


class GeometricMean(Statistic):
    def compute(self, data: List[float]):
        if any(x <= 0 for x in data):
            raise ValueError("Dados devem ser positivos para calcular a média geométrica")
        self.value = math.exp(sum(math.log(x) for x in data) / len(data))
        return self.value


# --- Classe Report ---
class Report:
    def __init__(self, statistics: List[Statistic]):
        self.statistics = statistics
        self.results = {}

    @staticmethod
    def normalize(data: List[float]) -> List[float]:
        min_val = min(data)
        max_val = max(data)
        epsilon = 1e-9
        if max_val - min_val == 0:
            return [epsilon for _ in data]
        normalized = [(x - min_val) / (max_val - min_val) for x in data]
        normalized = [max(x, epsilon) for x in normalized]
        return normalized

    def run(self, data: List[float], normalize_data: bool = False):
        if normalize_data:
            data = self.normalize(data)
        for stat in self.statistics:
            value = stat.compute(data)
            self.results[type(stat).__name__] = value

    def print_table(self):
        st.write("## Relatório de Estatísticas")
        table_data = {'Métrica': [], 'Valor': []}
        for metric, value in self.results.items():
            table_data['Métrica'].append(metric)
            table_data['Valor'].append(value)
        st.table(table_data)

    def plot_histogram(self, data: List[float], bins: int = 30):
        fig, ax = plt.subplots()
        ax.hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title('Histograma dos dados')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frequência')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)


# --- Função para ler CSV a partir de upload ---
def read_csv_from_upload(uploaded_file, column_name):
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    reader = csv.reader(stringio)
    headers = next(reader)
    if column_name not in headers:
        st.error(f"Coluna '{column_name}' não encontrada no CSV.")
        return []
    col_index = headers.index(column_name)
    data = []
    for row in reader:
        try:
            data.append(float(row[col_index]))
        except Exception:
            continue
    return data


# --- Função principal do app ---
def main():
    apply_custom_style()

    fonte = st.radio("Fonte de Dados:", ("Gerar dados aleatórios", "Importar CSV"))

    data = []

    if fonte == "Gerar dados aleatórios":
        n = st.number_input("Quantidade de dados (n)", min_value=10, max_value=100000, value=1000)
        low = st.number_input("Valor mínimo", value=1.0)
        high = st.number_input("Valor máximo", value=10.0)
        mode = st.selectbox("Distribuição", ["uniform", "normal"])
        seed = st.number_input("Semente de aleatoriedade", value=42)
        if st.button("Gerar Dados"):
            generator = DataGenerator(n, low, high, mode, seed)
            data[:] = generator.generate()
            st.success(f"Gerados {len(data)} dados.")

    else:
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])
        column_name = st.text_input("Nome da coluna numérica para análise")
        distribution_mode = st.selectbox("Distribuição para aplicar sobre dados importados", ["Nenhuma", "uniform", "normal"])
        if st.button("Importar CSV"):
            if uploaded_file is not None and column_name:
                data = read_csv_from_upload(uploaded_file, column_name)
                if data:
                    data_np = np.array(data)
                    if distribution_mode == "uniform":
                        # normaliza para 0..1
                        data_np = (data_np - data_np.min()) / (data_np.max() - data_np.min())
                        # escala para intervalo original
                        data_np = data_np * (data_np.max() - data_np.min()) + data_np.min()
                        data = list(data_np)
                    elif distribution_mode == "normal":
                        # transformar dados para ter média e desvio padrão da normal original carregada
                        mean = np.mean(data_np)
                        std = np.std(data_np)
                        data_np = np.random.normal(loc=mean, scale=std, size=len(data_np))
                        data = list(data_np)
                    st.success(f"Lidos {len(data)} valores da coluna '{column_name}' com distribuição '{distribution_mode}'.")

    if data:
        normalize_data = st.checkbox("Normalizar dados entre 0 e 1", value=True)

        metrics = [
            Mean(),
            Median(),
            StdDev(),
            MinValue(),
            MaxValue(),
            Variance(),
            Percentile25(),
            Percentile75(),
            Range(),
            GeometricMean()
        ]

        report = Report(metrics)
        try:
            report.run(data, normalize_data=normalize_data)
            report.print_table()
            report.plot_histogram(data)
        except ValueError as e:
            st.error(f"Erro ao calcular métricas: {e}")


if __name__ == "__main__":
    main()
