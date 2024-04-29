from collections import deque

def find_all_paths(graph, start, end, visited=[], path=[]):
    """
    This function finds all paths from a start node to an end node in a graph using DFS.

    Args:
        graph: A list of lists representing the adjacency matrix of the graph.
                graph[i][j] = 1 indicates an edge between node i and node j, 
                otherwise 0.
        start: The starting node.
        end: The ending node.
        visited: (internal use) Keeps track of visited nodes to avoid cycles.
        path: (internal use) Stores the current path during exploration.

    Returns:
        None (paths are appended to the 'all_paths' list outside the function)
    """
    visited.append(start)
    path.append(start)

    all_paths = []
    if start == end:
        # Found a path! Append it to the results
        all_paths.append(path.copy())
    else:
        # Explore unvisited neighbors
        for neighbor in range(len(graph[start])):
            if graph[start][neighbor] == 1 and neighbor not in visited:
                all_paths.extend(find_all_paths(graph, neighbor, end, visited.copy(), path.copy()))

    # Backtrack (remove current node from visited and path)
    visited.pop()
    path.pop()
    return all_paths

def shortest_path(graph, start, end):
    # Initialize a queue for BFS
    queue = deque()
    # Initialize a dictionary to store visited nodes and their parent node
    visited = {start: None}
    # Start BFS from the start node
    queue.append(start)

    # Perform BFS until the queue is empty
    while queue:
        node = queue.popleft()
        # If we reach the end node, reconstruct the path and return it
        if node == end:
            path = []
            while node is not None:
                path.append(node)
                node = visited[node]
            return path[::-1]

        # Explore neighbors of the current node
        for neighbor, connected in enumerate(graph[node]):
            if connected == 1 and neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)

    # If no path is found
    return None
