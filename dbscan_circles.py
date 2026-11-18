
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


# ==========================================================
# 1. CREAZIONE DEL DATASET
# ==========================================================

# Generiamo 400 punti distribuiti su due cerchi concentrici
X, y = make_circles(
    n_samples=400,       # Numero totale di punti
    noise=0.07,          # Quantità di rumore aggiunto ai punti
    factor=0.5,          # Dimensione relativa del cerchio interno
    random_state=42      # Rende ripetibile la generazione
)

# X contiene le coordinate dei punti
# y contiene le etichette originali dei due cerchi
# Attenzione: non utilizzeremo y per addestrare il clustering

print("Forma di X:", X.shape)
print("Forma di y:", y.shape)


# ==========================================================
# 2. STANDARDIZZAZIONE DEI DATI
# ==========================================================

# Creiamo lo standardizzatore
scaler = StandardScaler()

# Calcoliamo media e deviazione standard
# e trasformiamo le caratteristiche
X_std = scaler.fit_transform(X)


# ==========================================================
# 3. VISUALIZZAZIONE DEI DATI STANDARDIZZATI
# ==========================================================

plt.figure(figsize=(8, 6))

# Disegniamo tutti i punti utilizzando le due caratteristiche
plt.scatter(
    X_std[:, 0],       # Prima coordinata
    X_std[:, 1],       # Seconda coordinata
    s=20               # Dimensione dei punti
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Make Circles - Dati Standardizzati")

plt.grid(True)
plt.tight_layout()

# Salviamo il primo grafico in alta risoluzione
plt.savefig("make_circles_standardizzati.png",
            dpi=300, bbox_inches="tight")

plt.show()


# ==========================================================
# 4. CONFIGURAZIONE E ADDESTRAMENTO DBSCAN
# ==========================================================

# eps stabilisce il raggio di vicinanza tra i punti
# min_samples indica il numero minimo di punti nell'intorno
eps = 0.30
min_samples = 5

# Creiamo il modello DBSCAN
dbscan = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

# Addestriamo il modello sui dati standardizzati
dbscan.fit(X_std)

# Recuperiamo le etichette assegnate ai 400 punti
# Il valore -1 identifica i punti considerati rumore
labels = dbscan.labels_


# ==========================================================
# 5. ANALISI DEI CLUSTER E DEI PUNTI DI RUMORE
# ==========================================================

# Recuperiamo le etichette distinte
# Escludiamo -1 perché il rumore non è un cluster
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Contiamo i punti assegnati a un cluster
assigned = np.sum(labels != -1)

# Numero totale di osservazioni
tot = len(labels)

# Calcoliamo quanti punti sono stati classificati come rumore
noise = tot - assigned

# Calcoliamo la percentuale di punti assegnati
percentage = assigned / tot * 100

print("\n========== RISULTATI DBSCAN ==========")
print("Cluster individuati:", n_clusters)
print("Punti assegnati:", assigned)
print("Punti rumore:", noise)
print(f"Percentuale raggruppata: {percentage:.2f}%")


# ==========================================================
# 6. VISUALIZZAZIONE DEL CLUSTERING DBSCAN
# ==========================================================

# Recuperiamo tutte le etichette presenti
# Esempio: 0, 1, oppure -1 se esiste rumore
unique_labels = set(labels)

# Assegniamo un colore diverso a ogni etichetta
colors = [
    plt.cm.tab10(each)
    for each in np.linspace(0, 1, len(unique_labels))
]

plt.figure(figsize=(9, 7))

# Esaminiamo ogni cluster e l'eventuale gruppo di rumore
for k, col in zip(unique_labels, colors):

    # I punti di rumore vengono colorati di nero
    if k == -1:
        col = (0, 0, 0, 1)

    # Selezioniamo i punti che hanno questa specifica etichetta
    class_member_mask = (labels == k)

    # Recuperiamo le coordinate dei punti selezionati
    xy = X_std[class_member_mask]

    # Disegniamo i punti del gruppo corrente
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        c=[col],
        label=f"Cluster {k}" if k != -1 else "Rumore"
    )

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title(
    f"DBSCAN Clustering\n"
    f"eps = {eps} | min_samples = {min_samples}"
)

plt.legend()
plt.grid()
plt.tight_layout()

# Salviamo il grafico DBSCAN
plt.savefig("dbscan_clustering.png",
            dpi=300, bbox_inches="tight")

plt.show()


# ==========================================================
# 7. ADDESTRAMENTO K-MEANS PER IL CONFRONTO
# ==========================================================

# Creiamo K-Means chiedendo di individuare due gruppi
kmeans = KMeans(
    n_clusters=2,
    init="k-means++",
    n_init=10,
    random_state=42
)

# Addestriamo K-Means e assegniamo ogni punto a un cluster
clusters = kmeans.fit_predict(X_std)


# ==========================================================
# 8. VISUALIZZAZIONE DEL CLUSTERING K-MEANS
# ==========================================================

plt.figure(figsize=(9, 7))

# Disegniamo i punti colorandoli secondo i cluster trovati
scatter = plt.scatter(
    X_std[:, 0],
    X_std[:, 1],
    c=clusters,
    cmap="viridis",
    s=30,
    alpha=0.8
)

# Recuperiamo automaticamente i simboli dei due gruppi
handles, _ = scatter.legend_elements()

# Creiamo la legenda
plt.legend(
    handles,
    ["Cluster 0", "Cluster 1"],
    title="Gruppi individuati"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering - Make Circles")

plt.grid()
plt.tight_layout()

# Salviamo il grafico K-Means
plt.savefig("kmeans_clustering.png",
            dpi=300, bbox_inches="tight")

plt.show()
