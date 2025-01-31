import numpy as np
import pandas as pd
from IPython.display import clear_output
import math
import matplotlib.pyplot as plt
#from numpy import linalg as LA

df = pd.read_csv('full_randomized_2017_hourly_normalized_v2.csv') #Load demand and solar generation per kW dataframe

def obtain_dataid_2017(df):
    "Return all dataid as a nparray of a dataframe"
    return df[df.localhour == '2017-01-01 00:00:00']['dataid'].values.tolist()


dataids = obtain_dataid_2017(df) #list of dataids of firms
dataids_array = np.asarray(dataids) #numpy array of dataids 
validdata = len(dataids) #Total of firms
T = len(df[df.dataid==26].use.values) #Total of time slots
localtime_data = df[df.dataid==26].localhour #Hour data of time slots

def f_gen_per_kw(df):
    "Return gen_per_kw as a matrix on which rows are timeslots and columns are firms"
    dataids = obtain_dataid_2017(df)
    validdata = len(dataids)
    T = len(df[df.dataid==26].use.values)
    gen_kw = np.empty([T,validdata])
    for idx, val in enumerate(dataids):
        gen_kw[:,idx] = df[df.dataid == val]['gen_per_kW'].values
    return gen_kw

def f_load_kw(df):
    "Return gen_per_kw as a matrix on which rows are timeslots and columns are firms"
    dataids = obtain_dataid_2017(df)
    validdata = len(dataids)
    T = len(df[df.dataid==26].use.values)
    load_kw = np.empty([T,validdata])
    for idx, val in enumerate(dataids):
        load_kw[:,idx] = df[df.dataid == val]['use'].values
    return load_kw


gen_kw = f_gen_per_kw(df) #Matrix of normalized solar generation: Rows are timeslots and columns are firms
load_kw = f_load_kw(df) #Matrix of load: Rows are timeslots and columns are firms
gen_per_m2 = gen_kw*0.3/1.6354
#Investment per m2 is 512.2 $/m2.
#Using an anualized cost with discount rate r=5%, we obtain
CostPVperM2 = 512.2
r = 0.05
years = 20
annuity = CostPVperM2*r/(1-(1+r)**(-years)) #Obtain Annuity 
pi_s = annuity/T #Obtain cost per time step
print('Cost of PV per time step: ' + str(pi_s) + ' in $/m2')
#pi_o = 0.001
exp_lambda = 1

firms = validdata #Number of firms
gamma = 1 #Factor of NM price
pi_r = 0.18   #Retail Price at 18 cents per kWh.
pi_nm = gamma*pi_r  #Net Metering Price

def f_cap_firms(gen_per_m2, load_kw):
    "Return the annual cap of firms"
    avgPVUsers = np.mean(gen_per_m2, axis=0) #Obtain average generation per m2 of each home
    avgLoadUsers = np.mean(load_kw, axis=0) #Obtain average demand in kW of each home
    a_cap_firms = avgLoadUsers/avgPVUsers  #Obtain cap on investment to not be a producer through the year
    return a_cap_firms

a_cap_firms = f_cap_firms(gen_per_m2, load_kw) #Cap for firms to not be net producers


def f_lmp_prices(df):
    pi_g = df.MW.values #prices in $/MWh
    return pi_g

a_max_firms = 0.2*a_cap_firms
pv_changes = []
#gamma = np.linspace(0,1,21)


def solve_sharing_collective(gen_per_m2, load_kw, valid_index, a_max_size_firms, pi_s, pi_r, pi_o):
    #Data initialization
    T = len(gen_per_m2)
    avgPVFirms = np.mean(gen_per_m2, axis=0) #Obtain average generation per m2 per firm
    firms = len(a_max_size_firms)
    MaxGenFirms = avgPVFirms*a_max_size_firms #Obtain Max possible generation per firm
    idx_SortFirms = np.argsort(-MaxGenFirms) #Sort firms from highest maximum possible generation to lowest
    #dataid_SortFirms = valid_index[idx_SortFirms] #Obtain dataid of firms, sorted from highest to lowest max possible generation
    mid_firm = int(np.floor(firms/2)) #Divide by two the total number of firms, necessary to initialize the algorithm.
    
    #Initialize sets
    SetInvest = idx_SortFirms[0:mid_firm] #Set S that invest on PV. Initialized by picking the highest firms (half of them). Contain the index position used in MaxGenFirms or a_max_size_firms (and not dataid)
    SetNonInvest = idx_SortFirms[mid_firm:] #Set T initialized on PV. Complement of S. Contain the index used in MaxGenFirms or a_max_size_firms (and not dataid)
    SetInvest = SetInvest.tolist() #Convert array to list
    SetNonInvest = SetNonInvest.tolist() #Convert array to list
    
    Condition = True #Condition to stop the algorithm
    t = 1 #Iteration counter
    threshold = 5 #Threshold of how many repeated changes I admit
    counter = 0 #Counter of how many times the same change has been done

    #Initial values to start when the algorithm should stop
    dataid_S_old_1 = -2
    dataid_T_old_1 = -3
    dataid_removed_S = -4
    dataid_removed_T = -5
    
    while Condition:
        clear_output()
        print('\n')
        print('Iteration: ' + str(t))
        ## Compute Statistics of Collective PV and Load
        gen_max = gen_per_m2[:,SetInvest]*a_max_size_firms[SetInvest] #Users on S invest its max capacity
        collective_gen = np.sum(gen_max,axis=1) #collective gen in kW per time step
        collective_load = np.sum(load_kw,axis=1) #collective load in kW per time step
        net_load_pos = collective_load >= collective_gen #Timesteps when there is positive netload
        probdeficit = np.mean(net_load_pos) #Probability of deficit
        fault_dist = np.mean(np.random.uniform(low=0, high=1, size=len(net_load_pos)))
        operational = (pi_o/(probdeficit*pi_r)) - math.exp(-exp_lambda)*(pi_o/(probdeficit*pi_r))*fault_dist
        theta = pi_s/(probdeficit*pi_r) + operational #threshold for users
        #theta = pi_s/(probdeficit*pi_r)
        

        ## Compute Statistics of users regarding their merit site
        net_load_pos_vec = np.reshape(net_load_pos, (T,1)) #reshape vector of when net_load is positive
        W_pos = net_load_pos_vec*gen_per_m2 #Create vector of generation only when demand is positive
        expected_W_pos = np.sum(W_pos,axis=0)/np.sum(net_load_pos_vec) #Compute expected generation when netload is pos.
        merit_site_S = expected_W_pos[SetInvest] #Compute the merit of households on S
        merit_site_T = expected_W_pos[SetNonInvest] #Compute the merit of households on T
        firms_remove_from_S = merit_site_S < theta #If the merit site on S is below theta, they can be removed from S
        firms_remove_from_T = merit_site_T > theta #If the merit on site T is above theta, they can be removed from T

        aux_remove_S = -1 #Initialization of auxiliar variable to remove firm from S
        aux_remove_T = -1 #Initialization of auxiliar variable to remove firm from T


        if sum(firms_remove_from_S) > 0: #If there are firms to remove from S
            merit_site_S[~firms_remove_from_S]= float('Inf') #Set merit of non-removable firms to infinity
            aux_remove_S = np.argmin(merit_site_S) #Remove firm with the worst merit (to add to T)
            dataid_removed_S = SetInvest[aux_remove_S] #Save the index and dataid of the removed firm
            print('firm removed from S: '+ str(dataid_removed_S) + ' with dataid '+ str(valid_index[dataid_removed_S]))

        if sum(firms_remove_from_T) > 0: #If there are firms to remove from T
            merit_site_T[~firms_remove_from_T]= -float('Inf') #Set merit of non-removable firms to -infinity
            aux_remove_T = np.argmax(merit_site_T) #Remove firm with the best merit (to add to S)
            dataid_removed_T = SetNonInvest[aux_remove_T] #Save the index and dataid of the removed firm
            print('firm removed from T: '+ str(dataid_removed_T) + ' with dataid '+ str(valid_index[dataid_removed_T]))

        if aux_remove_S != -1: #If we have something to remove from S
            SetInvest.pop(aux_remove_S) #Remove the worst firm from set S
            SetNonInvest.append(dataid_removed_S) #Add worst firm removed from S to set T

        if aux_remove_T != -1: #If we have something to remove from T
            SetNonInvest.pop(aux_remove_T) #Remove the best firm from set T
            SetInvest.append(dataid_removed_T) #Add the best firm removed from T to set S
        if t==1:
            print('Set of firms that invest:')
            print(SetInvest)
            print('Set of firms that do not invest:')
            print(SetNonInvest)
        t = t+1
        if aux_remove_S + aux_remove_T == -2: #If no need to remove:
            Condition = False #End algorithm
        if dataid_T_old_1 == dataid_removed_S | dataid_S_old_1 == dataid_removed_T: #There is a repetition of removal and addition:
            counter = counter + 1
        if counter >= threshold: #If there is more than threshold repetition
            Condition = False #End algorithm
        dataid_T_old_1 = dataid_removed_T
        dataid_S_old_1 = dataid_removed_S
        print('Number of times doing the same change: '+ str(counter))
    print('\nTotal investment of PV in sharing case is ' + str(sum(a_max_size_firms[SetInvest])) + ' in m2')
    return SetInvest#Return which firms invest its maximum
    
def firms_investment_sharing(sol_sharing, a_max_size_firms, firms): #sol_sharing is what it is returned from the sharing algorithm
    "Return the investment of each firm for the sharing case"
    investment_pv = np.zeros(firms)
    investment_pv[sol_sharing] = a_max_size_firms[sol_sharing]
    return investment_pv


pi_o=[0.0 , 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
#pi_o=[0.0 , 0.00001, 0.00005]
invest_share=[]
sol_sharing_array=[]
for i in range(len(pi_o)):
    #Run the sharing case (that is not affected by gamma)
    soluc_sharing = solve_sharing_collective(gen_per_m2, load_kw, dataids_array, a_max_firms, pi_s, pi_r,pi_o[i]) #Obtain the set of firms that invest
    sol_sharing = firms_investment_sharing(soluc_sharing, a_max_firms, firms) #Obtain the investment decision of all firms
    invest_share.append(np.sum(sol_sharing))
    sol_sharing_array.append(sol_sharing)
    #clear_output() #Comment this line to check the intermediate steps of the algorithm



def solve_standalone(gen_per_m2, load_kw, valid_index, a_max_firms, pi_s, pi_r, pi_nm, pi_o):
    num_points = 500
    firms = len(a_max_firms)
    J_firms = np.zeros([num_points,firms])
    standalone_pv_m2 = np.zeros(firms)
    for i in range(firms):
        clear_output()
        print('Solving firm '+ str(i+1)+ ' (of '+ str(firms) +') with dataid ' + str(valid_index[i]))
        a = np.linspace(0,a_max_firms[i], num_points) #Create a vector between 0 and a_max.
        for k in range(len(a)):
            net_load = load_kw[:,i] - gen_per_m2[:,i]*a[k] #Compute net load for investment a[k]
            net_gen = -net_load
            net_load[net_load<=0] = 0 #Set to zero if the net load is negative (gen)
            net_gen[net_gen <=0] = 0 #Set to zero if the net gen is negative (load)
            J = pi_s*a[k] + np.mean(pi_r * net_load) - np.mean(pi_nm * net_gen) + a[k]*np.mean((1/exp_lambda)*pi_o*np.mean(np.random.uniform(low=0, high=1, size=(num_points))))
            J_firms[k,i] = J
        standalone_pv_m2[i] = a[np.argmin(J_firms[:,i])] #Save investment decision
    return standalone_pv_m2

invest_stand=[]
invest_standalone_array=[]
for i in range(len(pi_o)):
    pi_nm = gamma*pi_r
    investment_standalone = solve_standalone(gen_per_m2, load_kw, dataids, a_max_firms, pi_s, pi_r, pi_nm, pi_o[i])
    invest_stand.append(np.sum(investment_standalone))
    invest_standalone_array.append(investment_standalone)

diff_share_stand=[]
for a,b in zip(invest_share,invest_stand):
    #aux = (np.sum(sol_sharing) - np.sum(investment_standalone))/np.sum(investment_standalone)
    diff_share_stand.append((a-b)/b)
    #pv_changes.append(aux)
    #plt.plot(investment_standalone)
#clear_output()

nn = diff_share_stand
nn1 = [0.0,-0.006295647813559077,-0.021231983194252728,-0.11639765227269636,-0.4851854697306148,-0.7969210993681662,nn[7],nn[7]]
#nn1 for lambda = 1
plt.plot(pi_o,diff_share_stand)
plt.xlabel("Operational Price")
plt.ylabel("alpha")
plt.title("change in investment")
plt.legend()



plt.plot(pi_o, nn, color = 'r', label = "mean number of disasters = 2")
plt.plot(pi_o, nn1, color = 'b',label = "mean number of disasters = 1")
plt.xlabel("Operational Price")
plt.ylabel("alpha")
plt.title("mean number of disasters value 1 and 2")
plt.legend()

#Plot pv changes between standalone and sharing case
#plt.plot(gamma, pv_changes)
plt.plot(pi_o, invest_share, color = 'r', label = "sharing")
plt.plot(pi_o, invest_stand, color = 'b',label = "standalone")
plt.xlabel("Operational Price")
plt.ylabel("Total Investment of PV in square meter")
plt.title("Standalone and Sharing")
plt.legend()

plt.plot(a_max_firms)

print('\n')
print('Investment decisions:')
print(investment_standalone) #Print investment decisions
print('\n')
print('Percentage of investment per firm (with respect to max_cap)')
print(1-(a_max_firms - investment_standalone)/a_max_firms) #Print percentage of investment of available max cap
print('\n Total Investment of PV in standalone case is '+ str(sum(investment_standalone)) + ' in m2 \n')


def utility_profit_standalone(gen_per_m2, load_kw, standalone_inv, pi_r, pi_nm, pi_o):
    "Return the average profit per time slot when there is investment in PV using the standalone model"
    firms = len(standalone_inv)
    profit = 0
    for i in range(firms):
        net_load = load_kw[:,i] - gen_per_m2[:,i]*standalone_inv[i] #Compute net load for investment a[k]
        net_gen = -net_load
        net_load[net_load<=0] = 0 #Set to zero if the net load is negative (gen)
        net_gen[net_gen <=0] = 0 #Set to zero if the net gen is negative (load)
        profit = profit + np.mean(pi_r * net_load) - np.mean(pi_nm * net_gen) + standalone_inv[i]*np.mean((1/exp_lambda)*pi_o*np.mean(np.random.uniform(low=0, high=1, size=500))) #Receive pi_r if net load is positive and pay pi_nm if its negative
    print("Average profit per time slot for utility (with standalone PV investment) is " + str(profit))
    return profit

def utility_profit_sharing(gen_per_m2, load_kw, sharing_inv, pi_r, pi_o):
    "Return the average profit per time slot when there is investment in PV using the sharing model"
    profit = 0
    T = len(gen_per_m2)
    for t in range(T):
        InstalledGen = gen_per_m2[t,:]*sharing_inv
        CollectiveGen = np.sum(InstalledGen)
        CollectiveLoad = np.sum(load_kw[t,:])
        if CollectiveLoad > CollectiveGen:
            profit = profit + pi_r * (CollectiveLoad - CollectiveGen) + np.sum(sharing_inv*np.mean((1-math.exp(-exp_lambda))*pi_o*np.mean(np.random.uniform(low=0, high=1, size=500))))
    avg_profit = profit/T
    print("Average profit per time slot for utility (with sharing PV investment) is " + str(avg_profit))
    return avg_profit


utility_share=[]
utility_stand=[]
for i in range(len(pi_o)):
    #utility_stand.append(utility_profit_standalone(gen_per_m2, load_kw, investment_standalone, pi_r, pi_nm, pi_o[i]))
    utility_stand.append(utility_profit_standalone(gen_per_m2, load_kw, invest_standalone_array[i], pi_r, pi_nm, pi_o[i]))
    #utility_share.append(utility_profit_sharing(gen_per_m2, load_kw, sol_sharing, pi_r, pi_o[i]))
    utility_share.append(utility_profit_sharing(gen_per_m2, load_kw, sol_sharing_array[i], pi_r, pi_o[i]))


plt.plot(pi_o, utility_share, color = 'r', label = "sharing")
plt.plot(pi_o, utility_stand, color = 'b',label = "standalone")
plt.xlabel("Operational Price")
plt.ylabel("Profit due to PV investment in square meter")
plt.title("Standalone and Sharing")
plt.legend()


m=list(sol_sharing_array[5])
m=np.array(m)
non_zero_indices = np.nonzero(m)
x_values = non_zero_indices[0]
y_values = m[non_zero_indices]

m1= np.array(list(invest_standalone_array[5]))
non_zero_indices1 = np.nonzero(m1)
x_values1 = non_zero_indices1[0]
y_values1 = m1[non_zero_indices1]
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label="Sharing")
plt.xlabel("Households Id")
plt.ylabel("Area of PV invested in square meter")
plt.title("For Operational Price $0.0005 units per square meter")
plt.legend()
plt.plot(x_values1, y_values1, marker='o', linestyle='-', color='r', label="standalone")

#plt.plot(invest_standalone_array[4], color = 'b',label = "standalone")
plt.xlabel("Households Id")
plt.ylabel("Area of PV invested in square meter")
plt.title("For Operational Price $0.0005 units per square meter")
plt.legend()

###############################################

exp_lambda=[0.1, 1, 2, 5]
pi_o=0.005

def solve_sharing_collective(gen_per_m2, load_kw, valid_index, a_max_size_firms, pi_s, pi_r, pi_o, lambda_val):
    #Data initialization
    T = len(gen_per_m2)
    avgPVFirms = np.mean(gen_per_m2, axis=0) #Obtain average generation per m2 per firm
    firms = len(a_max_size_firms)
    MaxGenFirms = avgPVFirms*a_max_size_firms #Obtain Max possible generation per firm
    idx_SortFirms = np.argsort(-MaxGenFirms) #Sort firms from highest maximum possible generation to lowest
    #dataid_SortFirms = valid_index[idx_SortFirms] #Obtain dataid of firms, sorted from highest to lowest max possible generation
    mid_firm = int(np.floor(firms/2)) #Divide by two the total number of firms, necessary to initialize the algorithm.
    
    #Initialize sets
    SetInvest = idx_SortFirms[0:mid_firm] #Set S that invest on PV. Initialized by picking the highest firms (half of them). Contain the index position used in MaxGenFirms or a_max_size_firms (and not dataid)
    SetNonInvest = idx_SortFirms[mid_firm:] #Set T initialized on PV. Complement of S. Contain the index used in MaxGenFirms or a_max_size_firms (and not dataid)
    SetInvest = SetInvest.tolist() #Convert array to list
    SetNonInvest = SetNonInvest.tolist() #Convert array to list
    
    Condition = True #Condition to stop the algorithm
    t = 1 #Iteration counter
    threshold = 5 #Threshold of how many repeated changes I admit
    counter = 0 #Counter of how many times the same change has been done

    #Initial values to start when the algorithm should stop
    dataid_S_old_1 = -2
    dataid_T_old_1 = -3
    dataid_removed_S = -4
    dataid_removed_T = -5
    
    while Condition:
        clear_output()
        print('\n')
        print('Iteration: ' + str(t))
        ## Compute Statistics of Collective PV and Load
        gen_max = gen_per_m2[:,SetInvest]*a_max_size_firms[SetInvest] #Users on S invest its max capacity
        collective_gen = np.sum(gen_max,axis=1) #collective gen in kW per time step
        collective_load = np.sum(load_kw,axis=1) #collective load in kW per time step
        net_load_pos = collective_load >= collective_gen #Timesteps when there is positive netload
        probdeficit = np.mean(net_load_pos) #Probability of deficit
        fault_dist = np.mean(np.random.uniform(low=0, high=1, size=len(net_load_pos)))
        operational = (pi_o/(probdeficit*pi_r)) - math.exp(-lambda_val)*(pi_o/(probdeficit*pi_r))*fault_dist
        theta = pi_s/(probdeficit*pi_r) + operational #threshold for users
        #theta = pi_s/(probdeficit*pi_r)
        

        ## Compute Statistics of users regarding their merit site
        net_load_pos_vec = np.reshape(net_load_pos, (T,1)) #reshape vector of when net_load is positive
        W_pos = net_load_pos_vec*gen_per_m2 #Create vector of generation only when demand is positive
        expected_W_pos = np.sum(W_pos,axis=0)/np.sum(net_load_pos_vec) #Compute expected generation when netload is pos.
        merit_site_S = expected_W_pos[SetInvest] #Compute the merit of households on S
        merit_site_T = expected_W_pos[SetNonInvest] #Compute the merit of households on T
        firms_remove_from_S = merit_site_S < theta #If the merit site on S is below theta, they can be removed from S
        firms_remove_from_T = merit_site_T > theta #If the merit on site T is above theta, they can be removed from T

        aux_remove_S = -1 #Initialization of auxiliar variable to remove firm from S
        aux_remove_T = -1 #Initialization of auxiliar variable to remove firm from T


        if sum(firms_remove_from_S) > 0: #If there are firms to remove from S
            merit_site_S[~firms_remove_from_S]= float('Inf') #Set merit of non-removable firms to infinity
            aux_remove_S = np.argmin(merit_site_S) #Remove firm with the worst merit (to add to T)
            dataid_removed_S = SetInvest[aux_remove_S] #Save the index and dataid of the removed firm
            print('firm removed from S: '+ str(dataid_removed_S) + ' with dataid '+ str(valid_index[dataid_removed_S]))

        if sum(firms_remove_from_T) > 0: #If there are firms to remove from T
            merit_site_T[~firms_remove_from_T]= -float('Inf') #Set merit of non-removable firms to -infinity
            aux_remove_T = np.argmax(merit_site_T) #Remove firm with the best merit (to add to S)
            dataid_removed_T = SetNonInvest[aux_remove_T] #Save the index and dataid of the removed firm
            print('firm removed from T: '+ str(dataid_removed_T) + ' with dataid '+ str(valid_index[dataid_removed_T]))

        if aux_remove_S != -1: #If we have something to remove from S
            SetInvest.pop(aux_remove_S) #Remove the worst firm from set S
            SetNonInvest.append(dataid_removed_S) #Add worst firm removed from S to set T

        if aux_remove_T != -1: #If we have something to remove from T
            SetNonInvest.pop(aux_remove_T) #Remove the best firm from set T
            SetInvest.append(dataid_removed_T) #Add the best firm removed from T to set S
        if t==1:
            print('Set of firms that invest:')
            print(SetInvest)
            print('Set of firms that do not invest:')
            print(SetNonInvest)
        t = t+1
        if aux_remove_S + aux_remove_T == -2: #If no need to remove:
            Condition = False #End algorithm
        if dataid_T_old_1 == dataid_removed_S | dataid_S_old_1 == dataid_removed_T: #There is a repetition of removal and addition:
            counter = counter + 1
        if counter >= threshold: #If there is more than threshold repetition
            Condition = False #End algorithm
        dataid_T_old_1 = dataid_removed_T
        dataid_S_old_1 = dataid_removed_S
        print('Number of times doing the same change: '+ str(counter))
    print('\nTotal investment of PV in sharing case is ' + str(sum(a_max_size_firms[SetInvest])) + ' in m2')
    return SetInvest#Return which firms invest its maximum
    
def firms_investment_sharing(sol_sharing, a_max_size_firms, firms): #sol_sharing is what it is returned from the sharing algorithm
    "Return the investment of each firm for the sharing case"
    investment_pv = np.zeros(firms)
    investment_pv[sol_sharing] = a_max_size_firms[sol_sharing]
    return investment_pv



#pi_o=[0.0 , 0.00001, 0.00005]
invest_share_lambda=[]
sol_sharing_array_lambda=[]
for i in range(len(exp_lambda)):
    #Run the sharing case (that is not affected by gamma)
    soluc_sharing_lambda = solve_sharing_collective(gen_per_m2, load_kw, dataids_array, a_max_firms, pi_s, pi_r,pi_o,exp_lambda[i]) #Obtain the set of firms that invest
    sol_sharing_lambda = firms_investment_sharing(soluc_sharing_lambda, a_max_firms, firms) #Obtain the investment decision of all firms
    invest_share_lambda.append(np.sum(sol_sharing_lambda))
    sol_sharing_array_lambda.append(sol_sharing_lambda)
    #clear_output() #Comment this line to check the intermediate steps of the algorithm



def solve_standalone(gen_per_m2, load_kw, valid_index, a_max_firms, pi_s, pi_r, pi_nm, pi_o, lambda_val):
    num_points = 500
    firms = len(a_max_firms)
    J_firms = np.zeros([num_points,firms])
    standalone_pv_m2 = np.zeros(firms)
    for i in range(firms):
        clear_output()
        print('Solving firm '+ str(i+1)+ ' (of '+ str(firms) +') with dataid ' + str(valid_index[i]))
        a = np.linspace(0,a_max_firms[i], num_points) #Create a vector between 0 and a_max.
        for k in range(len(a)):
            net_load = load_kw[:,i] - gen_per_m2[:,i]*a[k] #Compute net load for investment a[k]
            net_gen = -net_load
            net_load[net_load<=0] = 0 #Set to zero if the net load is negative (gen)
            net_gen[net_gen <=0] = 0 #Set to zero if the net gen is negative (load)
            J = pi_s*a[k] + np.mean(pi_r * net_load) - np.mean(pi_nm * net_gen) + a[k]*np.mean((1/lambda_val)*pi_o*np.mean(np.random.uniform(low=0, high=1, size=(num_points))))
            J_firms[k,i] = J
        standalone_pv_m2[i] = a[np.argmin(J_firms[:,i])] #Save investment decision
    return standalone_pv_m2

invest_stand_lambda=[]
invest_standalone_array_lambda=[]
for i in range(len(exp_lambda)):
    pi_nm = gamma*pi_r
    investment_standalone_lambda = solve_standalone(gen_per_m2, load_kw, dataids, a_max_firms, pi_s, pi_r, pi_nm, pi_o, exp_lambda[i])
    invest_stand_lambda.append(np.sum(investment_standalone_lambda))
    invest_standalone_array_lambda.append(investment_standalone_lambda)

diff_share_stand_lambda=[]
for a,b in zip(invest_share_lambda,invest_stand_lambda):
    #aux = (np.sum(sol_sharing) - np.sum(investment_standalone))/np.sum(investment_standalone)
    diff_share_stand_lambda.append((a-b)/b)
    #pv_changes.append(aux)
    #plt.plot(investment_standalone)
#clear_output()

plt.plot(exp_lambda,diff_share_stand_lambda)
plt.xlabel("lambda")
plt.ylabel("alpha")
plt.title("change in investment")
plt.legend()

nn = diff_share_stand_lambda
nn1 = [0.08537519215134291,-0.00622734923484874,-0.016392768627893347,-0.023059408642682546]
#nn1 for lambda = 1
plt.plot(exp_lambda, nn, color = 'r', label = "operational price = 0.0001 per square meter")
plt.plot(exp_lambda, nn1, color = 'b',label = "operational price = 0.00001 per square meter")
plt.xlabel("lambda")
plt.ylabel("alpha")
plt.title("operational price value 0.0001 and 0.00001 per square meter")
plt.legend()

#Plot pv changes between standalone and sharing case
#plt.plot(gamma, pv_changes)
plt.plot(exp_lambda, invest_share_lambda, color = 'r', label = "sharing")
plt.plot(exp_lambda, invest_stand_lambda, color = 'b',label = "standalone")
plt.xlabel("lambda")
plt.ylabel("Total Investment of PV in square meter")
plt.title("Standalone and Sharing")
plt.legend()

plt.plot(a_max_firms)

print('\n')
print('Investment decisions:')
print(investment_standalone) #Print investment decisions
print('\n')
print('Percentage of investment per firm (with respect to max_cap)')
print(1-(a_max_firms - investment_standalone)/a_max_firms) #Print percentage of investment of available max cap
print('\n Total Investment of PV in standalone case is '+ str(sum(investment_standalone)) + ' in m2 \n')


def utility_profit_standalone(gen_per_m2, load_kw, standalone_inv, pi_r, pi_nm, pi_o,lambda_val):
    "Return the average profit per time slot when there is investment in PV using the standalone model"
    firms = len(standalone_inv)
    profit = 0
    for i in range(firms):
        net_load = load_kw[:,i] - gen_per_m2[:,i]*standalone_inv[i] #Compute net load for investment a[k]
        net_gen = -net_load
        net_load[net_load<=0] = 0 #Set to zero if the net load is negative (gen)
        net_gen[net_gen <=0] = 0 #Set to zero if the net gen is negative (load)
        profit = profit + np.mean(pi_r * net_load) - np.mean(pi_nm * net_gen) + standalone_inv[i]*np.mean((1/lambda_val)*pi_o*np.mean(np.random.uniform(low=0, high=1, size=500))) #Receive pi_r if net load is positive and pay pi_nm if its negative
    print("Average profit per time slot for utility (with standalone PV investment) is " + str(profit))
    return profit

def utility_profit_sharing(gen_per_m2, load_kw, sharing_inv, pi_r, pi_o, lambda_val):
    "Return the average profit per time slot when there is investment in PV using the sharing model"
    profit = 0
    T = len(gen_per_m2)
    for t in range(T):
        InstalledGen = gen_per_m2[t,:]*sharing_inv
        CollectiveGen = np.sum(InstalledGen)
        CollectiveLoad = np.sum(load_kw[t,:])
        if CollectiveLoad > CollectiveGen:
            profit = profit + pi_r * (CollectiveLoad - CollectiveGen) + np.sum(sharing_inv*np.mean((1-math.exp(-lambda_val))*pi_o*np.mean(np.random.uniform(low=0, high=1, size=500))))
    avg_profit = profit/T
    print("Average profit per time slot for utility (with sharing PV investment) is " + str(avg_profit))
    return avg_profit


utility_share_lambda=[]
utility_stand_lambda=[]
for i in range(len(exp_lambda)):
    #utility_stand.append(utility_profit_standalone(gen_per_m2, load_kw, investment_standalone, pi_r, pi_nm, pi_o[i]))
    utility_stand_lambda.append(utility_profit_standalone(gen_per_m2, load_kw, invest_standalone_array[i], pi_r, pi_nm, pi_o, exp_lambda[i]))
    #utility_share.append(utility_profit_sharing(gen_per_m2, load_kw, sol_sharing, pi_r, pi_o[i]))
    utility_share_lambda.append(utility_profit_sharing(gen_per_m2, load_kw, sol_sharing_array[i], pi_r, pi_o, exp_lambda[i]))

plt.plot(exp_lambda, utility_share_lambda, color = 'r', label = "sharing")
plt.plot(exp_lambda, utility_stand_lambda, color = 'b',label = "standalone")
plt.xlabel("Lambda")
plt.ylabel("Profit due to PV investment in square meter")
plt.title("Standalone and Sharing")
plt.legend()

plt.plot(sol_sharing_array_lambda[3], color = 'r', label = "sharing")
plt.plot(invest_standalone_array_lambda[3], color = 'b',label = "standalone")
plt.xlabel("Households")
plt.ylabel("Are of PV invested in square meter")
plt.title("For lambda 5")
plt.legend()


n=list(sol_sharing_array_lambda[0])
n=np.array(n)
non_zero_indices = np.nonzero(n)
x_values = non_zero_indices[0]
y_values = n[non_zero_indices]

n1= np.array(list(invest_standalone_array_lambda[0]))
non_zero_indices1 = np.nonzero(n1)
x_values1 = non_zero_indices1[0]
y_values1 = n1[non_zero_indices1]
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label="Sharing")
plt.xlabel("Households Id")
plt.ylabel("Area of PV invested in square meter")
plt.title("For lambda 1")
plt.legend()
plt.plot(x_values1, y_values1, marker='o', linestyle='-', color='r', label="standalone")

#plt.plot(invest_standalone_array[4], color = 'b',label = "standalone")
plt.xlabel("Households Id")
plt.ylabel("Area of PV invested in square meter")
plt.title("For lambda 1")
plt.legend()
