import matplotlib.pyplot as plt  # Para la visualización gráfica
import networkx as nx  # Para crear y manipular el grafo

# Función para encontrar el Árbol de Expansión Mínima usando el algoritmo de Kruskal
def kruskal(graph, reverse=False):
    # Ordenar aristas por peso en orden ascendente (o descendente si reverse=True)
    edges = [(weight, u, v) for u in graph for v, weight in graph[u].items()]
    edges.sort(reverse=reverse)

    parent = {i: i for i in graph}  # Inicializar el padre de cada nodo a sí mismo
    rank = {i: 0 for i in graph}  # Inicializar el rango de cada nodo a 0

    def find(u):
        # Encuentra la raíz del conjunto al que pertenece el nodo u
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        # Unir dos conjuntos
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            # Unir por rango
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_u] = root_v
                if rank[root_u] == rank[root_v]:
                    rank[root_v] += 1

    mst_edges = []
    for weight, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst_edges.append((u, v, weight))

    return mst_edges

# Grafo representando una red eléctrica con nodos (estaciones) y pesos (costes de cableado)
graph = {
    'A': {'B': 4, 'H': 8},
    'B': {'A': 4, 'H': 11, 'C': 8},
    'H': {'A': 8, 'B': 11, 'I': 7, 'G': 1},
    'C': {'B': 8, 'I': 2, 'F': 4, 'D': 7},
    'I': {'H': 7, 'C': 2, 'G': 6},
    'G': {'H': 1, 'I': 6, 'F': 2},
    'F': {'C': 4, 'G': 2, 'D': 14},
    'D': {'C': 7, 'F': 14, 'E': 9},
    'E': {'D': 9}
}

# Calcular el Árbol de Expansión Mínima (MST) usando el algoritmo de Kruskal
mst = kruskal(graph, reverse=False)
# Calcular el Árbol de Expansión Máxima (Máximo Coste) usando el algoritmo de Kruskal
mst_max = kruskal(graph, reverse=True)

# Imprimir el MST calculado
print("Árbol de Expansión Mínima (MST):")
for u, v, weight in mst:
    print(f"Desde {u} hasta {v} con peso {weight}")

# Imprimir el Árbol de Expansión Máxima calculado
print("\nÁrbol de Expansión Máxima:")
for u, v, weight in mst_max:
    print(f"Desde {u} hasta {v} con peso {weight}")

# Visualizar el grafo original, el MST y el Árbol de Expansión Máxima
plt.figure(figsize=(18, 6))

# Grafo Original
plt.subplot(131)
G = nx.Graph()
for node in graph:
    for neighbor, weight in graph[node].items():
        G.add_edge(node, neighbor, weight=weight)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title('Grafo Original')

# Árbol de Expansión Mínima (MST)
plt.subplot(132)
MST = nx.Graph()
for u, v, weight in mst:
    MST.add_edge(u, v, weight=weight)
pos_mst = nx.spring_layout(MST)
nx.draw(MST, pos_mst, with_labels=True, node_color='lightgreen', node_size=500, font_size=10)
nx.draw_networkx_edge_labels(MST, pos_mst, edge_labels={(u, v): weight for u, v, weight in mst})
plt.title('Árbol de Expansión Mínima (MST)')

# Árbol de Expansión Máxima
plt.subplot(133)
MST_MAX = nx.Graph()
for u, v, weight in mst_max:
    MST_MAX.add_edge(u, v, weight=weight)
pos_mst_max = nx.spring_layout(MST_MAX)
nx.draw(MST_MAX, pos_mst_max, with_labels=True, node_color='lightcoral', node_size=500, font_size=10)
nx.draw_networkx_edge_labels(MST_MAX, pos_mst_max, edge_labels={(u, v): weight for u, v, weight in mst_max})
plt.title('Árbol de Expansión Máxima')

# Mostrar las gráficas
plt.tight_layout()
plt.show()
