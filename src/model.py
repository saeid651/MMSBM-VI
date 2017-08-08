import numpy as np
import pymc

def create(data_matrix, num_people, num_groups, alpha, B):
    data_vector = data_matrix.reshape(num_people*num_people).T

    pi_list = np.empty(num_people, dtype=object)
    for person in range(num_people):
        person_pi = pymc.Dirichlet('pi_%i' % person, theta=alpha)
        pi_list[person] = person_pi

    z_pTq_matrix = np.empty([num_people,num_people], dtype=object)
    z_pFq_matrix = np.empty([num_people,num_people], dtype=object)
    for p_person in range(num_people):
        for q_person in range(num_people):
            z_pTq_matrix[p_person,q_person] = pymc.Multinomial('z_%dT%d_vector' % (p_person,q_person), n=1, p=pi_list[p_person])
            z_pFq_matrix[p_person,q_person] = pymc.Multinomial('z_%dF%d_vector' % (p_person,q_person), n=1, p=pi_list[q_person])

    @pymc.deterministic
    def bernoulli_parameters(z_pTq=z_pTq_matrix, z_pFq=z_pFq_matrix, B=B):
        bernoulli_parameters = np.empty([num_people, num_people], dtype=object)
        for p in range(num_people):
            for q in range(num_people):
                bernoulli_parameters[p,q] = np.dot(np.dot(z_pTq[p,q], B), z_pFq[p,q])
        return bernoulli_parameters.reshape(1,num_people*num_people)

    y_vector = pymc.Bernoulli('y_vector', p=bernoulli_parameters, value=data_vector, observed=True)

    return locals()
