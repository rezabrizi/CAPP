import os
import re
from collections import defaultdict, deque

index = ""
refs = []
year = ""
citations = defaultdict(lambda: {'refs': [], 'year': None})
with open("/Users/rezatabrizi/Downloads/.txt", "r") as f:
    file = f.read()
    file = file.split('#')
    
    for paper in file: 
        paper = paper.rstrip()
        if paper.startswith("index"):
            if(index != ""):
                citations[index]["refs"] = refs
                citations[index]["year"] = year
                refs = []

            index = int(re.sub(r'index(\d+)', r'\1', paper))

        elif paper.startswith("%"):
            ref = int(paper[1:])
            refs.append(ref)

        elif (paper.startswith("t")):
            year = int(paper[1:])
   
    if(index and refs and year):
        citations[index]["refs"]=refs
        citations[index]["year"] = year

graph = defaultdict(lambda : {"children": set(), "year": None})

for child_index, data in citations.items():
    graph[child_index]["year"] = data["year"]
    
    for parent_index in data["refs"]:
        graph[parent_index]["children"].add(child_index)

in_degree = defaultdict(int)

for u, data in graph.items():
    in_degree[u] += 0
    for v in data["children"]:
        in_degree[v] += 1




visited = defaultdict(int)
cascade = []
def dfs(dfs_u, t):
    visited[dfs_u] = 1

    for dfs_v in graph[dfs_u]["children"]:
        if visited[dfs_v] == 0: 
            dfs(dfs_v, t+1)
    
    visited[dfs_u] = 2
    cascade.append((dfs_u, t))



cascades = []
for node, ind in in_degree.items():
    if(ind) == 0:
        dfs(node, 0)
        cascade.reverse()
        cascades.append(cascade)
        visited = defaultdict(int)
        cascade = []

print(cascades)

with open("/Users/rezatabrizi/Downloads/citation_cas.txt", "w") as f: 
    for idx, c in enumerate(cascades):
        cas_string = str(idx) + " "
        for (node, activation) in c: 
            cas_string += str(node) + ':' + str(activation) + " "
        f.write(cas_string)
        f.write('\n')
        









        

        

        




