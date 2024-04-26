import copy

def recurse(graph, start, answer, path, input_start):
    """
    https://sexy-developer.tistory.com/m/55
    """
    path.append(start)
    cands = []

    # 갈 수 있는 노드 찾기
    for i in range(len(graph[start])):
        if i != start and graph[start][i] == 1:
            cands.append(i)

    # 갈 수 있는 노드가 없다면, 현재까지의 path를 answer의 원소로 저장하고, return하는 것으로 종료
    if not cands:
        answer.append(copy.deepcopy(path))
        return

    # 갈 수 있는 노드가 있다면, 그 노드로 가는 경로에 해당하는 graph의 원소를 0으로 바꾸고, 재귀 호출
    else:
        while cands:
            cand = cands.pop(0)
            graph[start][cand] = 0
            graph[cand][start] = 0
            recurse(graph, cand, answer, path, input_start)

        # 반복이 끝났다는 것은 해당 노드에서 더이상 갈 수 있는 노드가 없다는 뜻. 따라서 path에서 pop한다.
        path.pop()


def dfs_recursive(graph, input_start):
    path = []
    answer = []
    start = input_start
    recurse(graph, start, answer, path, input_start)

    return answer