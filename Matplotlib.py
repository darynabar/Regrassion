import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("graphs_report.pdf") as pdf:

    # Завдання 1
    x = np.linspace(-10, 10, 500)
    y = x**2 * np.sin(x)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="x²·sin(x)", color="blue")
    plt.title("Графік функції x²·sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    pdf.savefig()
    plt.close()

    # Завдання 2
    data = np.random.normal(loc=5, scale=2, size=1000)
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=30, color='orange', edgecolor='black')
    plt.title("Гістограма нормального розподілу (μ=5, σ=2)")
    plt.xlabel("Значення")
    plt.ylabel("Частота")
    plt.grid(True)
    pdf.savefig()
    plt.close()

    # Завдання 3
    hobbies = ['Програмування', 'Читання', 'Спорт', 'Музика', 'Подорожі']
    hours = [25, 20, 15, 20, 20]
    plt.figure(figsize=(6, 6))
    plt.pie(hours, labels=hobbies, autopct='%1.1f%%', startangle=140)
    plt.title("Мої хобі — розподіл часу")
    plt.axis('equal')
    pdf.savefig()
    plt.close()

    # Завдання 4
    fruits = ['Яблука', 'Банани', 'Апельсини', 'Груші']
    masses = [np.random.normal(loc=150, scale=20, size=100),
              np.random.normal(loc=120, scale=15, size=100),
              np.random.normal(loc=130, scale=18, size=100),
              np.random.normal(loc=140, scale=25, size=100)]
    plt.figure(figsize=(8, 5))
    plt.boxplot(masses, labels=fruits)
    plt.title("Box-plot маси фруктів (в грамах)")
    plt.ylabel("Маса (г)")
    plt.grid(True)
    pdf.savefig()
    plt.close()

