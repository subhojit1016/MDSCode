import time
import pandas as pd
import networkx as nx
import copy

# Start time to measure execution time
startTime = time.time()

# Function to compute the number of covered nodes

def covered(arr, node_arr, G):
     
    cov=[]
    for i in range(len(arr)):
        if (arr[i] == 1):
            index_ele = node_arr[i]
            list1 = [n for n in G.neighbors(index_ele)]
            cov = cov + list1
    cov = list(dict.fromkeys(cov))
    #print(cov)
    return (len(cov))
            
# Function to count occurrences of an element in a list
def countX(lst, x):
    return lst.count(x)       

# Objective function to evaluate the quality of the dominating set
def objective(DS_cover, arr, node_arr):
    if (sum(arr)==0):
        fx = (DS_cover/len(node_arr)) + 0.001
    else:
        fx = (DS_cover/len(node_arr)) + (1/(len(node_arr)*sum(arr)))
    return fx

# Alternative objective function considering only DS coverage
def objective1(DS_cover, node_arr):
    fx1 = (DS_cover/len(node_arr))
    return fx1

# Function to compute degree of nodes
def degree(arr, node_arr, G):
    deg = []
    for i in range(len(arr)):
        index_ele = node_arr[i]
        deg.append(G.degree[index_ele] - 2)
    #print(deg)
    return deg

# Function to find the next node to process
def node(ar1, ar2, ele, c):
    array_1 = copy.deepcopy(ar1)
    array_2 = copy.deepcopy(ar2)
    h = array_2.index(ele)
    f = array_1[h]
    if (ele in c):
        array_2.remove(ele)
        array_1.remove(f)
        f = min(array_1)
        e = array_1.index(f)
        ele = array_2[e]
    return ele
    
# Function to find the next highest element
def next_highest(Y1, y2):
    for i in range(len(Y1)):
        if (Y1[i]>y2):
            return Y1[i]
            break;
    
# Function to find the node with the minimum degree
def minimum_degree(array1, d, vertex, listed, G, var):
    degree_1 = []
    temp_array = []
    for i in range(len(array1)):
        if (array1[i] == 1):
            degree_1.append(d[i])
            temp_array.append(vertex[i])
    index_element = temp_array[var]
    return index_element
 
# Function to find the node with the maximum degree
def maximum_degree_later(array1, d, vertex, y):
    degree_1 = []
    temp_array = []
    for i in range(len(array1)):
        if (array1[i] == 0):
            degree_1.append(d[i])
            temp_array.append(vertex[i])
            
    val = max(degree_1)
    index = degree_1.index(val)
    index_element = temp_array[index]  
    return index_element    

# Function to refine the minimum dominating set
def fn(j, Y):
    dominating_set1=copy.deepcopy(Y)
    j = 0
    for i in range(j, len(dominating_set1)):
        if (dominating_set1[i] == 1):
            #element = potential_nodes[i]
            dominating_set1[i] = 0
            no = covered(dominating_set1, new_potent_nodes, G2)
            score = objective1(no, new_potent_nodes)
            if (score != 1):
                dominating_set1[i] = 1
                j=i+1
    return dominating_set1  
               

# Local search function to optimize the dominating set
def local_search(X1, node_arr, G, b, variable):
    degree_x1 = degree(X1, node_arr, G)
    covered_x1 = covered(X1, node_arr, G)
    objective_x1 = objective(covered_x1, X1, node_arr)
    if (objective_x1>1):        
        minimum = minimum_degree(X1, degree_x1, node_arr, b, G, variable)
        index_to_be_modified = node_arr.index(minimum)
        cnt=[]
        for k in range(len(X1)):
            if (X1[k] == 0):
                cnt.append(degree_x1[k])
        cnt1=[]
        for k in range(len(X1)):
            if (X1[k] == 1):
                cnt1.append(degree_x1[k])
        new_covered_x1 = covered(X1, node_arr, G)
        new_objective_x1 = objective(new_covered_x1, X1, node_arr)
        if (sum(X1)==0):
            X1[index_to_be_modified] = 1
            return X1,minimum,variable
        else:
            if (max(cnt) >= max(cnt1) and new_objective_x1<=1):
                X1[index_to_be_modified] = 1
                return X1,minimum,variable
            else:
                if(new_objective_x1 < objective_x1):
                    minimum_new = minimum_degree(X1, degree_x1, node_arr, b, G, variable)
                    index_to_be_modified_new = node_arr.index(minimum_new)
                    maximum_new = maximum_degree_later(X1, degree_x1, node_arr, minimum)
                    index_to_be_swapped = node_arr.index(maximum_new)
                    X1[index_to_be_swapped] = 1
                    X1[index_to_be_modified_new] = 0
                    new_covered_x2 = covered(X1, node_arr, G)
                    new_objective_x2 = objective(new_covered_x2, X1, node_arr)
                    if (new_objective_x2 > objective_x1):
                        return X1, minimum_new, variable+1
                    else:
                        X1[index_to_be_swapped] = 0
                        X1[index_to_be_modified_new] = 1
                        X1[index_to_be_modified] = 1
                        return X1, minimum_new, variable +1
                else:
                    return X1 , minimum, variable
           
        

# Read the CSV file
df = pd.read_csv ('C:/Users/subhojit.biswas/621_Project/621_Project_100a.csv')
print (df)
corr=df.to_numpy()


# Create a graph from adjacency matrix
G2 = nx.Graph()
nodes_i=[]
nodes_j=[]
for i in range(len(corr)):
    for j in range(len(corr[0])):
        if (corr[i][j]==1):
            nodes_i.append(i)
            nodes_j.append(j)
            
for i,j in zip (nodes_i,nodes_j):
    G2.add_edges_from([(j, i)])   
    

nx.draw_circular(G2, with_labels = True, node_size = 500)


# Compute minimum dominating set
F= nx.connected_components(G2)
Y = (list(F))
C=[]
for j in range(len(Y)):
    
    potent_nodes=list(Y[j])
    Z = [1 for i in range(len(potent_nodes))]
    print("The nodes"  +str(potent_nodes))
    deg=degree(Z, potent_nodes, G2)
    dg = copy.deepcopy(deg)
    deg.sort()
    new_potent_nodes=[]
    i=0 
    j=0       
    while j < len(deg):
        print("outerloop " + str(j))
        while i < len(dg):
            print ("Inner loop " + str(i))
            go=dg.index(dg[i])
            v=potent_nodes[go]
            print("str " + str(v))
            if (deg[j] == dg[i]):
                new_potent_nodes.append(v)
                deg.remove(deg[j])
                dg.remove(dg[i])
                print("dg " + str(dg))
                print("deg " + str(deg))
                potent_nodes.remove(v)
                print("potent_nodes " + str(potent_nodes))
                j=0
                i=0
            else:
                i = i +1
        
    print(new_potent_nodes)   
    if (len(new_potent_nodes) == 1):
        
        C=C+new_potent_nodes
    else:
        
        Z = [1 for i in range(len(new_potent_nodes))]
        c=[]
        counter_ele=0
        for k in range(1000):
            dominate , element ,counter_ele = local_search(Z,new_potent_nodes,G2, c,counter_ele)
            Z = dominate
            c.append(element)
            deg_X1 = degree(dominate, new_potent_nodes, G2)
            print("The node is " + str(element))
            print("The value is " + str(dominate))
            print("The nodes visited " + str(c)) 
            print("counter Element " + str(counter_ele))
            ar1 = []
            ar2 =[]
            for k in range(len(dominate)):
                if (dominate[k] == 0):
                    ar1.append(deg_X1[k])
            
            for k in range(len(dominate)):
                if (dominate[k] == 1):
                    ar2.append(deg_X1[k])
            
            if (max(ar1)>max(ar2)):
                break;
        minimum_dominating_set = fn(0, dominate)
        M=[]
        for i in range(len(minimum_dominating_set)):
            if (minimum_dominating_set[i] == 1):
                M.append(new_potent_nodes[i])
            
        C=C+M

print("The final MDS using local search is " + str(C))
print("The cardinality of minimum dominating set using local search is " + str(len(C)))
executionTime = (time.time() - startTime)
print('Execution time in seconds for local search: ' + str(executionTime))    


