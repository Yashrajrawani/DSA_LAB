from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import heapq

@dataclass
class Building:
    id: int
    name: str
    location: str

    def __repr__(self):
        return f"Building(id={self.id}, name='{self.name}', location='{self.location}')"



class BSTNode:
    def __init__(self, building: Building):
        self.building = building
        self.left: Optional['BSTNode'] = None
        self.right: Optional['BSTNode'] = None

class BinarySearchTree:
    def __init__(self):
        
        self.root: Optional[BSTNode] = None

    def insert(self, building: Building):
        if not self.root:
            self.root = BSTNode(building)
            return
        cur = self.root
        while True:
            if building.id < cur.building.id:
                if cur.left is None:
                    cur.left = BSTNode(building)
                    return
                cur = cur.left
            elif building.id > cur.building.id:
                if cur.right is None:
                    cur.right = BSTNode(building)
                    return
                cur = cur.right
            else:
                
                cur.building = building
                return

    def search(self, building_id: int) -> Optional[Building]:
        cur = self.root
        while cur:
            if building_id == cur.building.id:
                return cur.building
            elif building_id < cur.building.id:
                cur = cur.left
            else:
                cur = cur.right
        return None

    def _inorder(self, node: Optional[BSTNode], res: List[Building]):
        if node:
            self._inorder(node.left, res)
            res.append(node.building)
            self._inorder(node.right, res)

    def inorder(self) -> List[Building]:
        res: List[Building] = []
        self._inorder(self.root, res)
        return res

    def _preorder(self, node: Optional[BSTNode], res: List[Building]):
        if node:
            res.append(node.building)
            self._preorder(node.left, res)
            self._preorder(node.right, res)

    def preorder(self) -> List[Building]:
        res: List[Building] = []
        self._preorder(self.root, res)
        return res

    def _postorder(self, node: Optional[BSTNode], res: List[Building]):
        if node:
            self._postorder(node.left, res)
            self._postorder(node.right, res)
            res.append(node.building)

    def postorder(self) -> List[Building]:
        res: List[Building] = []
        self._postorder(self.root, res)
        return res

    def _height(self, node: Optional[BSTNode]) -> int:
        if not node:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))

    def height(self) -> int:
        return self._height(self.root)



class AVLNode:
    def __init__(self, building: Building):
        self.building = building
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
        self.height: int = 1

class AVLTree:
    def __init__(self):
        self.root: Optional[AVLNode] = None

    def _get_height(self, node: Optional[AVLNode]) -> int:
        return node.height if node else 0

    def _update_height(self, node: AVLNode):
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

    def _get_balance(self, node: Optional[AVLNode]) -> int:
        if not node: return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_right(self, y: AVLNode) -> AVLNode:
        x = y.left
        T2 = x.right
       
        x.right = y
        y.left = T2
      
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x: AVLNode) -> AVLNode:
        y = x.right
        T2 = y.left
       
        y.left = x
        x.right = T2
       
        self._update_height(x)
        self._update_height(y)
        return y

    def _insert(self, node: Optional[AVLNode], building: Building) -> AVLNode:
        if not node:
            return AVLNode(building)
        if building.id < node.building.id:
            node.left = self._insert(node.left, building)
        elif building.id > node.building.id:
            node.right = self._insert(node.right, building)
        else:
           
            node.building = building
            return node

        self._update_height(node)
        balance = self._get_balance(node)

        
        if balance > 1 and building.id < node.left.building.id:
            return self._rotate_right(node)
        
        if balance < -1 and building.id > node.right.building.id:
            return self._rotate_left(node)
       
        if balance > 1 and building.id > node.left.building.id:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
       
        if balance < -1 and building.id < node.right.building.id:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    def insert(self, building: Building):
        self.root = self._insert(self.root, building)

    def search(self, building_id: int) -> Optional[Building]:
        cur = self.root
        while cur:
            if building_id == cur.building.id:
                return cur.building
            elif building_id < cur.building.id:
                cur = cur.left
            else:
                cur = cur.right
        return None

    def _inorder(self, node: Optional[AVLNode], res: List[Building]):
        if node:
            self._inorder(node.left, res)
            res.append(node.building)
            self._inorder(node.right, res)

    def inorder(self) -> List[Building]:
        res: List[Building] = []
        self._inorder(self.root, res)
        return res

    def height(self) -> int:
        return self._get_height(self.root)



class ExprNode:
    def __init__(self, value: str):
        self.value = value
        self.left: Optional['ExprNode'] = None
        self.right: Optional['ExprNode'] = None

class ExpressionTree:
    @staticmethod
    def build_from_postfix(tokens: List[str]) -> Optional[ExprNode]:
        stack: List[ExprNode] = []
        ops = set(['+', '-', '*', '/', '^'])
        for t in tokens:
            if t not in ops:
                stack.append(ExprNode(t))
            else:
                # binary operator
                right = stack.pop()
                left = stack.pop()
                node = ExprNode(t)
                node.left = left
                node.right = right
                stack.append(node)
        return stack[-1] if stack else None

    @staticmethod
    def evaluate(node: Optional[ExprNode]) -> float:
        if node is None:
            return 0
        if node.left is None and node.right is None:
            
            return float(node.value)
        left_val = ExpressionTree.evaluate(node.left)
        right_val = ExpressionTree.evaluate(node.right)
        if node.value == '+': return left_val + right_val
        if node.value == '-': return left_val - right_val
        if node.value == '*': return left_val * right_val
        if node.value == '/': return left_val / right_val
        if node.value == '^': return left_val ** right_val
        raise ValueError(f"Unknown operator {node.value}")


class CampusGraph:
    def __init__(self, directed: bool=False):
        self.adj_list: Dict[int, List[Tuple[int, float]]] = {}  
        self.directed = directed

    def add_building(self, b: Building):
        self.nodes_info[b.id] = b
        self.adj_list.setdefault(b.id, [])

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        if u not in self.adj_list or v not in self.adj_list:
            raise KeyError("Both nodes must be added to graph before adding edges")
        self.adj_list[u].append((v, weight))
        if not self.directed:
            self.adj_list[v].append((u, weight))

    def adjacency_matrix(self) -> Tuple[List[int], List[List[float]]]:
        ids = sorted(self.adj_list.keys())
        n = len(ids)
        id_index = {node_id: i for i, node_id in enumerate(ids)}
        INF = float('inf')
        mat = [[INF]*n for _ in range(n)]
        for i in range(n):
            mat[i][i] = 0.0
        for u in ids:
            for v, w in self.adj_list[u]:
                i = id_index[u]; j = id_index[v]
                mat[i][j] = w
        return ids, mat

    def bfs(self, start_id: int) -> List[int]:
        visited = set()
        order = []
        queue = [start_id]
        visited.add(start_id)
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v, _ in self.adj_list.get(u, []):
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return order

    def dfs(self, start_id: int) -> List[int]:
        visited = set()
        order = []
        def _dfs(u):
            visited.add(u)
            order.append(u)
            for v, _ in self.adj_list.get(u, []):
                if v not in visited:
                    _dfs(v)
        _dfs(start_id)
        return order

    def dijkstra(self, src: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        dist = {node: float('inf') for node in self.adj_list}
        prev: Dict[int, Optional[int]] = {node: None for node in self.adj_list}
        dist[src] = 0.0
        heap = [(0.0, src)]
        while heap:
            d,u = heapq.heappop(heap)
            if d>dist[u]:
                continue
            for v,w in self.adj_list[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap,(nd,v))
        return dist, prev

    def shortest_path(self, src: int, dest: int) -> Tuple[float, List[int]]:
        dist, prev = self.dijkstra(src)
        if dist[dest] == float('inf'):
            return float('inf'), []
        path = []
        cur = dest
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return dist[dest], path

   
    def kruskal_mst(self) -> Tuple[float, List[Tuple[int,int,float]]]:
        
        edges = []
        seen = set()
        for u in self.adj_list:
            for v,w in self.adj_list[u]:
                if self.directed:
                    edges.append((w,u,v))
                else:
                    
                    if (v,u) in seen:
                        continue
                    edges.append((w,u,v))
                    seen.add((u,v))
        edges.sort(key=lambda x: x[0])
        parent = {node: node for node in self.adj_list}
        rank = {node: 0 for node in self.adj_list}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x,y):
            rx,ry = find(x), find(y)
            if rx==ry: return False
            if rank[rx] < rank[ry]:
                parent[rx]=ry
            elif rank[rx]>rank[ry]:
                parent[ry]=rx
            else:
                parent[ry]=rx
                rank[rx]+=1
            return True

        mst_edges = []
        total_cost = 0.0
        for w,u,v in edges:
            if union(u,v):
                mst_edges.append((u,v,w))
                total_cost += w
        return total_cost, mst_edges


def main():
    
    buildings = [
        Building(10, "Admin Block", "North Campus"),
        Building(5, "Library", "Central Campus"),
        Building(20, "CS Dept", "East Wing"),
        Building(3, "Cafeteria", "Ground Floor"),
        Building(7, "Auditorium", "Main Hall"),
        Building(15, "Hostel A", "North-West"),
    ]

    
    bst = BinarySearchTree()
    for b in buildings:
        bst.insert(b)
    print("--- BST Traversals ---")
    print("Inorder:", bst.inorder())
    print("Preorder:", bst.preorder())
    print("Postorder:", bst.postorder())
    print("BST Height:", bst.height())

    
    avl = AVLTree()
    for b in buildings:
        avl.insert(b)
    print("\n--- AVL Traversal & Height ---")
    print("AVL Inorder:", avl.inorder())
    print("AVL Height:", avl.height())

   
    print(f"\nHeight Comparison -> BST: {bst.height()}  |  AVL: {avl.height()}")

   
    g = CampusGraph(directed=False)
    for b in buildings:
        g.add_building(b)

    
    g.add_edge(10,5, 50.0)   
    g.add_edge(10,20, 120.0) 
    g.add_edge(5,3, 30.0)    
    g.add_edge(5,7, 60.0)    
    g.add_edge(7,20, 40.0)   
    g.add_edge(20,15, 80.0)  
    g.add_edge(3,15, 200.0)  

    print("\n--- Graph Representations ---")
    ids, mat = g.adjacency_matrix()
    print("Node IDs (ordered):", ids)
    print("Adjacency Matrix (rows/cols follow node order above):")
    for row in mat:
        
        print(["inf" if x==float('inf') else x for x in row])

    print("\n--- Graph Traversals ---")
    start = 10
    print("BFS from Admin (10):", g.bfs(start))
    print("DFS from Admin (10):", g.dfs(start))

    print("\n--- Shortest Path (Dijkstra) ---")
    dist, path = g.shortest_path(10, 15)
    print(f"Shortest distance from Admin(10) to Hostel A(15): {dist}")
    print("Path (building ids):", path)
    print("Path (names):", [g.nodes_info[i].name for i in path])

    print("\n--- Kruskal MST ---")
    total_cost, mst_edges = g.kruskal_mst()
    print("Total MST cost (distance):", total_cost)
    print("MST edges (u, v, w):", mst_edges)
    print("MST edges with names:", [(g.nodes_info[u].name, g.nodes_info[v].name, w) for (u,v,w) in mst_edges])

    print("\n--- Expression Tree (Energy bill calc) ---")
   
    postfix = ["200", "5", "*", "50", "+"]
    expr_root = ExpressionTree.build_from_postfix(postfix)
    value = ExpressionTree.evaluate(expr_root)
    print("Expression (units*rate + fixed):", value) 

   

if __name__ == "__main__":
    main()
