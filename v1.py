import pandas as pd
import numpy
from sklearn.neighbors import NearestNeighbors
import progressbar as pb


def organize_data_helper(df_to_organize):
    """
    The function organize the data in a way the will be easier to Analyze
    :param df_to_organize: a DataFrame
    :return: a new DataFrame
    """
    df_to_organize['user_id'] = range(1, len(df_to_organize) + 1)
    all_dfs = [
        df_to_organize[['user_id', f'c{item}', f'b{item}']].rename(columns={f'c{item}': 'Country', f'b{item}': 'Best'})
        for
        item in range(1, 6)]
    return pd.concat(all_dfs).sort_values(by='user_id').reset_index(drop=True)


def organize_data(df_to_organize):
    """
    Organizing the Data
    """
    df_to_organize = organize_data_helper(df_to_organize)
    df_to_organize['Best'] = df_to_organize['Best'].str.lower()
    df_to_organize['Country'] = df_to_organize['Country'].str.lower()
    return df_to_organize


def normalizer(df):
    """
    Normalize the Data to 1-5 rating
    """
    for col in df:
        max_old = df[col].max()
        min_old = df[col].min()
        df[col] = df[col].apply(
            lambda old_val: (4 / (max_old - min_old)) * (old_val - max_old) + 5)
    return df


def create_country_matrix(df):
    # Creating the Countries / Attributes matrix
    df_countries = pd.crosstab(df['Best'], df['Country'])
    return normalizer(df_countries)


def create_users_matrix(df):
    # Creating the Users / Attributes matrix
    df_users = pd.crosstab(df['user_id'], df['Best']).T
    return normalizer(df_users).T


def recommend_countries(df, df1, user, num_recommended_countries):
    # print('Since you already visited in:')
    # for country in df[df[user] > 0][user].index.tolist():
    #     print(country)
    recommended_countries = []
    for m in df[df[user] == 0].index.tolist():
        index_df = df.index.tolist().index(m)
        predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(user)]
        recommended_countries.append((m, predicted_rating))

    sorted_rm = sorted(recommended_countries, key=lambda x: x[1], reverse=True)
    print('The list of the Recommended Countries to Visit \n')
    rank = 1
    for recommended_country in sorted_rm[:num_recommended_countries]:
        print('{}: {} - predicted rating:{}'.format(rank, recommended_country[0], recommended_country[1]))
        rank = rank + 1


def matrix_factorization(R, P, Q, K, steps=500000, alpha=0.0002, beta=0.02):
    """
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Countries features matrix)
    K: len of attributes
    steps: iterations
    alpha: learning rate
    beta: regularization parameter
    """
    Q = Q.T
    for _ in pb.progressbar(range(steps)):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        print(e)
        if e < 0.001:
            break

    return P, Q.T


def nearest_neighbor(numpy_matrix, user, num_neighbors, num_recommendation):
    # Convert Back to DF
    matrix = pd.DataFrame(numpy_matrix, columns=ALL_COUNTRIES, index=ALL_USERS)
    df = matrix.T.copy()
    df1 = df.copy()
    number_neighbors = num_neighbors

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)

    user_index = df.columns.tolist().index(user)

    for m, t in list(enumerate(df.index)):
        if df.iloc[m, user_index] == 0:
            sim_countries = indices[m].tolist()
            countries_distances = distances[m].tolist()

            if m in sim_countries:
                id_country = sim_countries.index(m)
                sim_countries.remove(m)
                countries_distances.pop(id_country)

            else:
                sim_countries = sim_countries[:num_neighbors - 1]
                countries_distances = countries_distances[:num_neighbors - 1]

            country_similarity = [1 - x for x in countries_distances]
            country_similarity_copy = country_similarity.copy()
            nominator = 0

            for s in range(0, len(country_similarity)):
                if df.iloc[sim_countries[s], user_index] == 0:
                    if len(country_similarity_copy) == (number_neighbors - 1):
                        country_similarity_copy.pop(s)

                    else:
                        country_similarity_copy.pop(s - (len(country_similarity) - len(country_similarity_copy)))

                else:
                    nominator = nominator + country_similarity[s] * df.iloc[sim_countries[s], user_index]
            if len(country_similarity_copy) > 0:
                if sum(country_similarity_copy) > 0:
                    predicted_r = nominator / sum(country_similarity_copy)

                else:
                    predicted_r = 0

            else:
                predicted_r = 0

            df1.iloc[m, user_index] = predicted_r
    recommend_countries(df.copy(), df1.copy(), user, num_recommendation)


def recommender_system_algorithms():
    # N: num of Users
    N = len(numpy_matrix)
    # M: num of Countries
    M = len(numpy_matrix[0])
    # Num of Attributes
    K = len(df_users[0])
    # Filling the Matrix using Matrix Factorization
    P = numpy.random.rand(N, K)
    Q = numpy.random.rand(M, K)
    # nP, nQ = matrix_factorization(numpy_matrix, df_users, df_countries.T, K)
    nP, nQ = matrix_factorization(numpy_matrix, P, Q, K)

    nR = numpy.dot(nP, nQ.T)
    # Convert Back to DF
    matrix = pd.DataFrame(nR, columns=ALL_COUNTRIES, index=ALL_USERS)
    # matrix = normalizer(matrix)
    # Save to Excel
    matrix.to_excel('predictions.xlsx')


if __name__ == "__main__":
    # Loading the Data
    df = pd.read_excel('main.xlsx')
    df = organize_data(df)
    # Creating the Countries / Attributes matrix
    df_countries = create_country_matrix(df).values
    # Creating the Users / Attributes matrix
    df_users = create_users_matrix(df).values

    # Matrix Factorization
    numpy_matrix = numpy.dot(df_users, df_countries)
    # List of Countries
    ALL_COUNTRIES = df['Country'].unique().tolist()
    ALL_COUNTRIES.sort()
    # List of Users
    ALL_USERS = df['user_id'].unique().tolist()
    # Removing rating for users who hasn't visited in the country
    for i, row in enumerate(numpy_matrix):
        for j, col in enumerate(row):
            if ALL_COUNTRIES[j] not in df[df['user_id'] == i + 1]['Country'].to_list():
                numpy_matrix[i][j] = 0.0

    recommender_system_algorithms()
    nearest_neighbor(numpy_matrix, 6, 3, 5)
    # TODO: take matrix_factorization Excel and for each user recommend for a new 5 countries (5 max rating)
    #
