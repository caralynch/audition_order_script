import pandas as pd
import numpy as np
import sys
import os

pd.options.mode.chained_assignment = None

def make_fullname(row):
    """Create full name from first name and surname columns"""
    return f"{row.firstname} {row.surname}"

def swap_rows(df, row1, row2):
    df.iloc[row1], df.iloc[row2] =  df.iloc[row2].copy(), df.iloc[row1].copy()
    return df

# read in file from command argument
if len(sys.argv) < 2:
    print('No filename input after program name - assuming read from infile.txt')
    if os.path.exists('input.txt'):
        with open('input.txt', 'r') as infile:
            filename = str(infile.readline()).strip('\n')
    else:
        exit()
else:
    if len(sys.argv) > 2:
        print('Assuming only first argument is filename')
    filename = str(sys.argv[1])

# read in csv
data = pd.read_csv(filename)

# rename relevant columns
colnames = {
    'First Name': 'firstname',
    'Surname': 'surname',
    'Role 1': 'roles',
    'Available for Audition?': 'available',
    'Supporting Characters Preference': 'Supporting role preference',
    'Are you willing to accept a role you did not audition for?': 'accept non-auditioned role',
    'If you are not cast in a principal role, are you willing to be a member of the ensemble?': 'ensemble'
}
data = data.rename(columns=colnames)

# only want columns that interest us
data = data[colnames.values()]

# only need 1 name column
data['name'] = data.apply(make_fullname, axis=1)
data.drop(['firstname', 'surname'], axis=1, inplace=True)

# only include those available for auditions
data = data[data.available == 'Yes']

# separate out roles
data['roles'] = data.roles.str.split(', ')

# rename horrific role names 
rename_roles = {
    'Supporting Role Singing Audition (Busker Song)': 'Supporting Busker Song',
    'Supporting Role Singing Audition (Wedding Singer Song)': 'Supporting Wedding Song',
    'Supporting Role Acting Audition (either standard or Instructor)': 'Supporting Acting'
}
def rename_long_roles(role_list):
    for key in rename_roles:
        if key in role_list:
            role_list = list(map(lambda x: x.replace(key, rename_roles[key]), role_list))
    
    return role_list

data['roles'] = data.roles.apply(rename_long_roles)

data = data.reset_index(drop=True)

# save this iteration of the data to add more info at end
input_data = data.copy(deep=True)

# combine supporting roles to one role
def combine_supporting_roles(role_list):
    new_list = []
    for value in role_list:
        if value in rename_roles.values():
            if 'Supporting Role' not in new_list:
                new_list.append('Supporting Role')
        else:
            new_list.append(value)
    return new_list

data['roles'] = data.roles.apply(combine_supporting_roles)

# get role tuples
role_tuples = {}
for role_list in data.roles:
    if len(role_list) > 1:
        if len(role_list) == 2:
            new_tuples = [tuple(role_list)]
        else:
            new_tuples = []
            for i in range(0, len(role_list)):
                for j in range(i+1, len(role_list)):
                    new_tuples.append((role_list[i], role_list[j]))
        for new_tuple in new_tuples:
            if new_tuple in role_tuples:
                role_tuples[new_tuple] += 1
            elif (new_tuple[1], new_tuple[0]) in role_tuples:
                role_tuples[(new_tuple[1], new_tuple[0])] += 1
            else:
                role_tuples[new_tuple] = 1

# get pair order
pair_order = list(pd.DataFrame.from_dict(role_tuples, orient='index').sort_values(by=0, ascending=False).index)

# make running order dictionaries
running_order = {
    0: []
}
started = False
for pair in pair_order:
    i=0
    added = False
    while not added:
        if not started:
            running_order[i] = [pair[0], pair[1]]
            started = True
            added = True
        elif (pair[0] in running_order[i]) & (pair[1] in running_order[i]):
            added=True
        else:
            # if not in current running order
            if (pair[0] not in running_order[i]) & (pair[1] not in running_order[i]):
                # add new running order list if not one already
                if len(running_order) < i+2:
                    running_order[i+1] = [pair[0], pair[1]]
                    added = True
            # if one element is in current running order
            else:
                # find which element is in the running order
                if pair[0] in running_order[i]:
                    already_in = pair[0]
                    to_add = pair[1]
                elif pair[1] in running_order[i]:
                    already_in = pair[1]
                    to_add = pair[0]
                
                # add to end if element already present is in second half of running order
                if running_order[i].index(already_in) > len(running_order[i])/2:
                    running_order[i].append(to_add)
                else:
                    running_order[i].insert(0, to_add)
                
                added=True
        i+=1

# add extra roles if weren't in tuples
for role_list in data.roles:
    if len(role_list) < 2:
        role = role_list[0]
        found = False
        i = 0
        while not found:
            if role in running_order[i]:
                found = True
            else:
                if (i+1) in running_order.keys():
                    i += 1
                else:
                    running_order[i+1] = [role]
                    found = True

# get number of roles ppl have applied for
def get_role_number(value):
    return len(value)

data['role_nb'] = data.roles.apply(get_role_number)

# make a new row for each role somebody applied for        
new_data = data.copy(deep=True)
for i in data.index:
    roles = data.loc[i, 'roles']
    new_data.loc[i, 'roles'] = roles[0]
    if len(roles) > 1:
        to_add = pd.DataFrame([data.loc[i, :]*(len(roles)-1)]).reset_index(drop=True)
        for j in range(1, len(roles)):
            to_add.loc[j-1, 'roles'] = roles[j]
        
        new_data = pd.concat((new_data, to_add), ignore_index=True)
    
def get_row_distance(audition_order_df):
    """get dict of distances between rows for each person"""

    names = audition_order_df.name.unique()
    row_distances = {}
    for name in names:
        if len(audition_order_df[audition_order_df.name==name].index)>1:
            row_distance = audition_order_df[audition_order_df.name==name].index[1] - audition_order_df[audition_order_df.name==name].index[0]
        else:
            row_distance = 0
        row_distances[name] = row_distance
    return row_distances

def get_distance_sum(df):
    """get sum of distances between rows for each person"""
    return np.array(list(get_row_distance(df).values())).sum()

# make running orders
sorted_low_to_high = new_data.sort_values(by='role_nb')
sorted_high_to_low = new_data.sort_values(by='role_nb', ascending=False)
audition_lists = {}

for key in running_order:
    started = False
    for role in running_order[key]:
        if not started:
            audition_lists[key] = sorted_low_to_high[sorted_low_to_high.roles == role]
            started = True
        else:
            if running_order[key][-1] == role:
                audition_lists[key] = pd.concat((audition_lists[key], sorted_high_to_low[sorted_high_to_low.roles == role]), axis=0)
            else:
                # first get auditionees only auditioning for this role
                to_add  = new_data.loc[((new_data.roles == role) & (new_data.role_nb == 1)), :]

                # then deal with auditionees with 2 or more auditions
                double_auditionees = new_data.loc[((new_data.roles == role) & (new_data.role_nb > 1))]
                for auditionee in double_auditionees.name:
                    # if double auditionee already in list, add to top of audition group
                    if len(audition_lists[key][audition_lists[key].name == auditionee]) > 0:
                        to_add = pd.concat((double_auditionees[double_auditionees.name == auditionee], to_add))
                    else:
                        to_add = pd.concat((to_add, double_auditionees[double_auditionees.name == auditionee]))
                
                audition_lists[key] = pd.concat((audition_lists[key], to_add), axis=0)
        
    audition_lists[key] = audition_lists[key].reset_index()[['name', 'roles']].rename(columns={'roles': 'role'})

    if len(audition_lists[key]) > 2 & len(audition_lists[key].role.unique()) > 1:

        # check if there's a better order by 'rolling over' start and endpoints
        distance_sum_diff = 1
        while distance_sum_diff > 0:
            first_role_length = len(audition_lists[key].loc[audition_lists[key].role == audition_lists[key].loc[0, 'role'], :])
            first_new_df = pd.DataFrame(np.roll(audition_lists[key], shift=-first_role_length, axis=0), columns=audition_lists[key].columns)
            first_distance_sum = get_distance_sum(first_new_df)

            last_role_length = len(audition_lists[key].loc[audition_lists[key].role == audition_lists[key].loc[len(audition_lists[key])-1, 'role'], :])
            last_new_df = pd.DataFrame(np.roll(audition_lists[key], shift=last_role_length, axis=0), columns=audition_lists[key].columns)
            last_distance_sum = get_distance_sum(last_new_df)

            if last_distance_sum < first_distance_sum:
                new_distance_sum = last_distance_sum
                new_df = last_new_df
            else:
                new_distance_sum = first_distance_sum
                new_df = first_new_df
            
            distance_sum_diff = get_distance_sum(audition_lists[key]) - new_distance_sum
            if distance_sum_diff > 0:
                audition_lists[key] = new_df

        # make sure nobody has two consecutive auditions
        for i in audition_lists[key].index[:-1]:
            if audition_lists[key].loc[i, 'name'] == audition_lists[key].loc[i+1, 'name']:
                if i != len(audition_lists[key]) - 1:
                    audition_lists[key] = swap_rows(audition_lists[key], i+1, i+2)
                else:
                    audition_lists[key] = swap_rows(audition_lists[key], i-1, i)


# make nice output
def transform_roles_to_str(role_list):
    rolestr = ''
    for i, role in enumerate(role_list):
        if i != 0:
            rolestr += ', '
        rolestr += role

    return rolestr

input_data['all roles'] = input_data.roles.apply(transform_roles_to_str)

output_lists = {}
for key in audition_lists:
    output_lists[key] = audition_lists[key].merge(input_data[['name', 'all roles', 'Supporting role preference']], on='name', how='left')

outfile = 'audition_order'
if os.path.exists(f"{outfile}.xlsx"):
    version_number = 1
    while os.path.exists(f"{outfile}_v{version_number}.xlsx"):
        version_number += 1
    outfile = f"{outfile}_v{version_number}"
outfile = outfile + '.xlsx'
    

with pd.ExcelWriter(outfile, engine='xlsxwriter') as writer:
    for key in output_lists:
        output_lists[key].to_excel(writer, sheet_name=f"group_{key}", index=False)
        if len(output_lists[key]) > 1:
            output_lists[key].sort_index(ascending=False).to_excel(writer, sheet_name=f"group_{key}_inv", index=False)

