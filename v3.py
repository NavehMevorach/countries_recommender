import time
import sys
import numpy as np
import pandas as pd
import random
import json
import difflib


COUNTRIES = ['Afghanistan', 'Aland Islands', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla',
             'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan',
             'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda',
             'Bhutan', 'Bolivia, Plurinational State of', 'Bonaire, Sint Eustatius and Saba', 'Bosnia and Herzegovina',
             'Botswana', 'Bouvet Island', 'Brazil', 'British Indian Ocean Territory', 'Brunei Darussalam', 'Bulgaria',
             'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands',
             'Central African Republic', 'Chad', 'Chile', 'China', 'Christmas Island', 'Cocos (Keeling) Islands',
             'Colombia', 'Comoros', 'Congo', 'Congo, The Democratic Republic of the', 'Cook Islands', 'Costa Rica',
             "Côte d'Ivoire", 'Croatia', 'Cuba', 'Curaçao', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti',
             'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea',
             'Estonia', 'Ethiopia', 'Falkland Islands (Malvinas)', 'Faroe Islands', 'Fiji', 'Finland', 'France',
             'French Guiana', 'French Polynesia', 'French Southern Territories', 'Gabon', 'Gambia', 'Georgia',
             'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala',
             'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Heard Island and McDonald Islands',
             'Holy See (Vatican City State)', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia',
             'Iran, Islamic Republic of', 'Iraq', 'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan',
             'Jersey', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', "Korea, Democratic People's Republic of",
             'Korea, Republic of', 'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic", 'Latvia', 'Lebanon',
             'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macao',
             'Macedonia, Republic of', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta',
             'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius', 'Mayotte', 'Mexico',
             'Micronesia, Federated States of', 'Moldova, Republic of', 'Monaco', 'Mongolia', 'Montenegro',
             'Montserrat', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands',
             'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norfolk Island',
             'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestinian Territory, Occupied',
             'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn', 'Poland', 'Portugal',
             'Puerto Rico', 'Qatar', 'Réunion', 'Romania', 'Russian Federation', 'Rwanda', 'Saint Barthélemy',
             'Saint Helena, Ascension and Tristan da Cunha', 'Saint Kitts and Nevis', 'Saint Lucia',
             'Saint Martin (French part)', 'Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines', 'Samoa',
             'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone',
             'Singapore', 'Sint Maarten (Dutch part)', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia',
             'South Africa', 'South Georgia and the South Sandwich Islands', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname',
             'South Sudan', 'Svalbard and Jan Mayen', 'Swaziland', 'Sweden', 'Switzerland', 'Syrian Arab Republic',
             'Taiwan, Province of China', 'Tajikistan', 'Tanzania, United Republic of', 'Thailand', 'Timor-Leste',
             'Togo', 'Tokelau', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan',
             'Turks and Caicos Islands', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom',
             'United States', 'United States Minor Outlying Islands', 'Uruguay', 'Uzbekistan', 'Vanuatu',
             'Venezuela, Bolivarian Republic of', 'Viet Nam', 'Virgin Islands, British', 'Virgin Islands, U.S.',
             'Wallis and Futuna', 'Yemen', 'Zambia', 'Zimbabwe']


def delay_print(t, phase=90):
    typing_speed = phase  # wpm
    for l in t:
        sys.stdout.write(l)
        sys.stdout.flush()
        time.sleep(random.random() * 10.0 / typing_speed)
    print()


def check_similar_s(s):
    return difflib.get_close_matches(s, COUNTRIES)[0]


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def get_user_input(user_id):
    """
    Ask for the user input and return a list with all the data
    """
    OPTIONS = {
        1: 'Architecture',
        2: 'Art and Museums',
        3: 'Beach',
        4: 'Culture',
        5: 'Extreme Activities',
        6: 'Food',
        7: 'Hiking & Nature',
        8: 'History',
        9: 'Night Life',
        10: 'Shopping',
    }
    delay_print('Hello')
    delay_print("Welcome to Naveh Travel Recommendation Open-Source Project")
    delay_print("First, contribute by sharing with the community where have you traveled before")
    delay_print("And then, go ahead and enjoy the power of collaborative filtering =)")
    print()
    user_input = []
    delay_print("How many countries would you want to add to the Database?")
    amount_of_countries = int(input())
    print()
    delay_print("Awesome! let's start!")
    while amount_of_countries:
        delay_print("Please write the country name:")
        country_name = input()
        while country_name.lower() not in map(lambda x: x.lower(), COUNTRIES):
            delay_print("Its seems there isn't such country")
            delay_print("Did you meant to this one by any chance?")
            suggestion = check_similar_s(country_name)
            print(suggestion)
            answer = input("y/n")
            if answer == 'y':
                country_name = suggestion
            else:
                delay_print("Please write the country name again:")
                country_name = input()
        delay_print("What is the Main reason you visited this country?")
        print("(Pleas write the Number of the reason and not the reason itself)")
        print(json.dumps(OPTIONS, sort_keys=True, indent=4))
        attribute = int(input())
        user_input.append((user_id, country_name.lower(), OPTIONS[attribute].lower()))
        amount_of_countries -= 1
        if amount_of_countries:
            delay_print(f'Thanks! only {amount_of_countries} more to go \n')
        else:
            delay_print("That's it! would you want to get a recommendation?")
    get_recommendation = input("y/n ?")
    delay_print("Calculating...")
    delay_print("It might take while, meanwhile you should take a coffee break...")
    return user_input, get_recommendation


def update_raw_db(df, data):
    """
    Add the user input to the database with all the raw data
    """
    df2 = pd.DataFrame(data, columns=['user_id', 'Country', 'Best'])
    merged_df = pd.concat([df, df2], ignore_index=True, sort=False)
    merged_df.to_excel('raw_data.xlsx', index=False)
    return merged_df


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
    return df_countries
    # return normalizer(df_countries)


def create_users_matrix(df):
    # Creating the Users / Attributes matrix
    df_users = pd.crosstab(df['user_id'], df['Best']).T
    return df_users.T
    # return normalizer(df_users).T


def create_numpy_matrix(df):
    """
    Create the building block matrix that have all the ratings.
    The algorythm calculate all the raw data and gives a rating to each user based on community opinion for each country
    """
    # Creating the Countries / Attributes matrix
    df_countries = create_country_matrix(df)
    # Creating the Users / Attributes matrix
    df_users = create_users_matrix(df)
    return np.dot(df_users.values, df_countries.values), df_countries


def get_user_and_countries_list(df):
    # List of Countries
    ALL_COUNTRIES = df['Country'].unique().tolist()
    ALL_COUNTRIES.sort()
    # List of Users
    ALL_USERS = df['user_id'].unique().tolist()
    return ALL_USERS, ALL_COUNTRIES


def remove_unnecessary_rating(R, df, countries_list):
    """
    The rating is given only for countries the user actually visited at
    So this function convert the rating to 0 of everyone that arent
    """
    for i, row in enumerate(R):
        for j, col in enumerate(row):
            if countries_list[j] not in df[df['user_id'] == i + 1]['Country'].to_list():
                R[i][j] = 0
            else:
                R[i][j] = R[i][j] / 10


def create_matrix_building_block(R, df, users_list, countries_list):
    # Remove all rating of users that didn't actually visited the country
    remove_unnecessary_rating(R, df, countries_list)
    return pd.DataFrame(R, columns=countries_list, index=users_list).values


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


def suggest(df, m, id):
    countries_the_user_visited = (df[df['user_id'] == id]['Country'].to_list())
    # user_series_in_matrix = m.loc[id].sort_values().iloc[-(len(countries_the_user_visited) * 2 + 1):-1]
    user_series_in_matrix = m.loc[[id]].sort_values(by=id, axis=1, ascending=False)
    suggestions = []
    for i, name in enumerate(user_series_in_matrix):
        rating = user_series_in_matrix.iloc[:, i].to_list()[0]
        if name not in countries_the_user_visited:
            suggestions.append((name, rating))
        if len(suggestions) >= 3:
            break
    return suggestions


class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.
        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        # Initialize user and item latent feature matrix
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

    def train(self):
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * P_i - self.beta * self.Q[j, :])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)


if __name__ == "__main__":
    # TODO: If you dont have the Dataset you can uncomment the code below and import it from google sheets
    # Step 1 (Alternative) - Read the Data from the DB
    # gsheetid = "1HSIjDNlYd58u6IuLOwQkqvH4uRsxDNjYbLSRfXPDhfg"
    # sheet_name = "Sheet1"
    # gsheet_url = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(gsheetid, sheet_name)
    # df = pd.read_csv(gsheet_url)
    # Step 1 - Read the Data from the DB
    df = pd.read_excel('raw_data.xlsx')
    # Step 2 - Assign the User an Unique ID
    user_id = len(df['user_id'].unique()) + 1
    # Step 3 - Get input from the user
    input_to_add, wants_to_get_suggestions = get_user_input(user_id)
    # input_to_add = [(user_id, 'mexico', 'beach'),
    #                 (user_id, 'philippines', 'beach'),
    #                 (user_id, 'sri lanka', 'beach')]
    # wants_to_get_suggestions = True
    # Step 4 - Update the DB with the user input
    df = update_raw_db(df, input_to_add)
    # Step 5 - Create a Matrix from the DB with the Ratings of each user for each country
    R, df_countries = create_numpy_matrix(df)
    # Step 6 - Create two matrices for the Factorization Based on Masses Wisdom Algorithm
    users_list, countries_list = get_user_and_countries_list(df)
    # Step 7 - Create the Matrix Building Block for the Training Algorithm
    building_block_matrix = create_matrix_building_block(R, df, users_list, countries_list)
    # Step 8 - Training the Data and using Matrix Factorization for getting the Final Matrix
    mf = MF(R, K=10, alpha=0.1, beta=0.01, iterations=5000)
    training_process = mf.train()
    recommendation_matrix = mf.full_matrix()
    # Step 9 - Normalizing the Data again
    matrix = normalizer(pd.DataFrame(recommendation_matrix, columns=countries_list, index=users_list))
    # Step 10 (Optional) - Adding the Data to a new DB
    matrix.to_excel('predictions.xlsx')
    # Step 11 - Suggest the user what countries to go
    if wants_to_get_suggestions:
        suggestion_one, suggestion_two, suggestion_three = suggest(df, matrix, user_id)
        delay_print(f'Hurray! I found you a 3 places you should visit!')
        delay_print(f'Some Drums please')
        delay_print(f'🥁🥁🥁🥁🥁🥁🥁🥁🥁🥁', 20)
        delay_print(f'In the first place: \n {suggestion_one[0]} for {df_countries[suggestion_one[0]].sort_values().index[-1]}')
        delay_print(f'In the second place: \n {suggestion_two[0]} for {df_countries[suggestion_two[0]].sort_values().index[-1]}')
        delay_print(f'In the third place: \n {suggestion_three[0]} for {df_countries[suggestion_three[0]].sort_values().index[-1]}')

    delay_print("Thanks for contributing the project! see you next time!")



