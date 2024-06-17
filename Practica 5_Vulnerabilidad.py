import matplotlib.pyplot as plt  # Para la visualización gráfica
import networkx as nx  # Para crear y manipular el grafo

# Función para encontrar el Árbol de Expansión Mínima (MST) usando el algoritmo de Kruskal
def kruskal_minimum(graph):
    # Ordenar todas las aristas del grafo por peso en orden ascendente
    edges = sorted((weight, u, v) for u in graph for v, weight in graph[u].items())
    # Inicializar estructuras para union-find
    parent = {i: i for i in graph}
    rank = {i: 0 for i in graph}

    # Función para encontrar el representante de un conjunto
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    # Función para unir dos conjuntos
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_u] = root_v
                if rank[root_u] == rank[root_v]:
                    rank[root_v] += 1

    mst_edges = []  # Lista para almacenar las aristas del MST
    for weight, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst_edges.append((u, v, weight))

    return mst_edges

# Función para encontrar el Árbol de Expansión Máxima usando el algoritmo de Kruskal
def kruskal_maximum(graph):
    # Ordenar todas las aristas del grafo por peso en orden descendente
    edges = sorted(((weight, u, v) for u in graph for v, weight in graph[u].items()), reverse=True)
    # Inicializar estructuras para union-find
    parent = {i: i for i in graph}
    rank = {i: 0 for i in graph}

    # Función para encontrar el representante de un conjunto
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    # Función para unir dos conjuntos
    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_u] = root_v
                if rank[root_u] == rank[root_v]:
                    rank[root_v] += 1

    mst_edges = []  # Lista para almacenar las aristas del MST
    for weight, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst_edges.append((u, v, weight))

    return mst_edges

# Grafo representando una red con nodos y pesos
graph = {
    'A': {'B': 4, 'C': 8},
    'B': {'A': 4, 'C': 11, 'D': 8},
    'C': {'A': 8, 'B': 11, 'D': 2, 'E': 7},
    'D': {'B': 8, 'C': 2, 'E': 6},
    'E': {'C': 7, 'D': 6}
}

# Calcular el Árbol de Expansión Mínima (MST) usando el algoritmo de Kruskal
mst_min = kruskal_minimum(graph)

# Calcular el Árbol de Expansión Máxima usando el algoritmo de Kruskal
mst_max = kruskal_maximum(graph)

# Crear un grafo con NetworkX para calcular rutas
G_min = nx.Graph()
for u, v, weight in mst_min:
    G_min.add_edge(u, v, weight=weight)

G_max = nx.Graph()
for u, v, weight in mst_max:
    G_max.add_edge(u, v, weight=weight)

# Imprimir el MST de mínimo coste y el MST de máximo coste
print("Árbol de Expansión Mínima (MST) - Coste Mínimo:")
for u, v, weight in mst_min:
    print(f"Desde {u} hasta {v} con peso {weight}")

print("\nÁrbol de Expansión Máxima - Coste Máximo:")
for u, v, weight in mst_max:
    print(f"Desde {u} hasta {v} con peso {weight}")

# Visualizar el grafo original, el MST de mínimo coste y el MST de máximo coste
plt.figure(figsize=(18, 6))

# Grafo Original
plt.subplot(131)
G_original = nx.Graph()
for node in graph:
    for neighbor, weight in graph[node].items():
        G_original.add_edge(node, neighbor, weight=weight)
pos_original = nx.spring_layout(G_original)
nx.draw(G_original, pos_original, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
labels_original = nx.get_edge_attributes(G_original, 'weight')
nx.draw_networkx_edge_labels(G_original, pos_original, edge_labels=labels_original)
plt.title('Grafo Original')

# Árbol de Expansión Mínima (MST)
plt.subplot(132)
pos_min = nx.spring_layout(G_min)
nx.draw(G_min, pos_min, with_labels=True, node_color='lightgreen', node_size=500, font_size=10)
labels_min = nx.get_edge_attributes(G_min, 'weight')
nx.draw_networkx_edge_labels(G_min, pos_min, edge_labels=labels_min)
plt.title('Árbol de Expansión Mínima (MST) - Coste Mínimo')

# Árbol de Expansión Máxima
plt.subplot(133)
pos_max = nx.spring_layout(G_max)
nx.draw(G_max, pos_max, with_labels=True, node_color='lightcoral', node_size=500, font_size=10)
labels_max = nx.get_edge_attributes(G_max, 'weight')
nx.draw_networkx_edge_labels(G_max, pos_max, edge_labels=labels_max)
plt.title('Árbol de Expansión Máxima - Coste Máximo')

# Mostrar las gráficas
plt.tight_layout()
plt.show()
