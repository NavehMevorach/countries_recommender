import pandas as pd
import numpy
from itertools import groupby


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


def organize_data(df_to_organize):
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


def check_countries_names():
    """
    Get a list of countries name and return the names that a re valid
    :return: list with all the countries names that were valid
    """
    pass


def check_amount_of_similar_countries(countries, user):
    """
    Get a list of countries and check how many are inside the c1
    :param countries:
    :param user:
    :return:
    """
    countries = list(map(lambda c: c.lower(), countries.keys()))
    similar = []
    for country in user.Country:
        if country.lower() in countries:
            similar.append(country)
    return similar


def find_similarity(user_input_df, df):
    """
    The function get the User Input which contain 3 countries names
    - Check if the countries name are Valid
    - Iterate through the user and give score by amount of similarity
    :return: a Series with user_id as index and amount of similar countries as value
    """
    user_rating_series = df.groupby('user_id').apply(
        lambda user: check_amount_of_similar_countries(user_input_df, user))
    return user_rating_series[user_rating_series.apply(len) > 0]


def calc_rating(amount_of_countries, amount_of_attr):
    """
    Return the rating based on how strong the similarity is
    :return: int which stand for the similarity strength
    """
    if amount_of_countries == 3 and amount_of_attr == 3: return 9
    if amount_of_countries == 3 and amount_of_attr == 2: return 8
    if amount_of_countries == 2 and amount_of_attr == 2: return 7
    if amount_of_countries == 3 and amount_of_attr == 1: return 6
    if amount_of_countries == 2 and amount_of_attr == 1: return 5
    if amount_of_countries == 3 and amount_of_attr == 0: return 4
    if amount_of_countries == 2 and amount_of_attr == 0: return 3
    if amount_of_countries == 1 and amount_of_attr == 1: return 2
    if amount_of_countries == 1 and amount_of_attr == 0: return 1


def rate_users(df, countries, series):
    """
    The function get a Series of all the users with similar countries
    - The function will split them into 3 groups  based of the amont of similar countries
    - The function Rate each user based of amount of similar attributes
    :return: Series with the Rating of each user
    """
    input_countries = list(countries.keys())
    for user, value in series.items():
        rating = 0
        for country in value:
            attr = df[(df['user_id'] == user) & (df['Country'] == country)].Best.iloc[0]
            if countries[country.lower()] == attr.lower():
                rating += 1
        final_rating = calc_rating(len(value), rating)
        series[user] = final_rating
    return series


def updating_list_of_options(row, lst, user_rating_series):
    country = row.iloc[1]
    user_id = row.user_id
    rating = user_rating_series.loc[user_id]
    lst += [country] * rating


def most_common(lst):
    most_common_country = max(set(lst), key=lst.count)
    lst = list(filter(lambda x: x != most_common_country, lst))
    second_most_common_country = max(set(lst), key=lst.count)
    return most_common_country, second_most_common_country


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def suggest_what_todo_helper(df1, df2, attr1, attr2):
    df1_rating_for_attr1 = 0
    df1_rating_for_attr2 = 0
    df2_rating_for_attr1 = 0
    df2_rating_for_attr2 = 0

    if attr1 in df1.columns:
        df1_rating_for_attr1 = df1[attr1].iloc[0]
    if attr2 in df1.columns:
        df1_rating_for_attr2 = df1[attr2].iloc[0]
    if attr1 in df2.columns:
        df2_rating_for_attr1 = df2[attr1].iloc[0]
    if attr2 in df2.columns:
        df2_rating_for_attr2 = df2[attr2].iloc[0]

    what_todo_in_country_one = attr1 if df1_rating_for_attr1 >= df1_rating_for_attr2 else attr2
    what_todo_in_country_two = attr1 if df2_rating_for_attr1 >= df2_rating_for_attr2 else attr2

    return what_todo_in_country_one, what_todo_in_country_two


def suggest_what_todo(test_countries, country_one, country_two):
    countries_activities_rating = pd.read_excel('summarized.xlsx')
    countries_activities_rating.country = countries_activities_rating.country.apply(lambda x: x.lower())
    suggested_country_one_data = countries_activities_rating[countries_activities_rating['country'] == country_one]
    suggested_country_two_data = countries_activities_rating[countries_activities_rating['country'] == country_two]

    things_user_like_todo = list(test_countries.values())
    favorite_thing_todo = ''
    second_favorite_thing_todo = ''
    if all_equal(things_user_like_todo):
        favorite_thing_todo = things_user_like_todo[0]
    else:
        favorite_thing_todo, second_favorite_thing_todo = most_common(things_user_like_todo)

    return suggest_what_todo_helper(suggested_country_one_data, suggested_country_two_data, favorite_thing_todo,
                                    second_favorite_thing_todo)


def main(df):
    # Organize the Data
    df = organize_data(df)
    df['Country'] = df['Country'].apply(lambda x: x.lower())
    df['Best'] = df['Best'].apply(lambda x: x.lower())
    # Find users with similarities of choices
    user_with_similarity_series = find_similarity(test_countries, df)
    # Rate the similar users by the similarity of Attributes
    user_rating_series = rate_users(df, test_countries, user_with_similarity_series)
    # Filtered the Main Dataframe so there only be relevant users
    filtered_df = df[df['user_id'].isin(user_rating_series.index)]
    filtered_df = filtered_df[~filtered_df['Country'].isin(test_countries.keys())]
    # Creating a list of all relevant countries by length
    filtered_df.apply(lambda row: updating_list_of_options(row, results, user_rating_series), axis='columns')
    # Print the Two Most relevant countries
    country_one, country_two = most_common(results)
    print(f"Your two next countries you should visit are: {country_one.capitalize()} and {country_two.capitalize()}")
    # Suggest what to do in those countries
    what_todo_in_country_one, what_todo_in_country_two = suggest_what_todo(test_countries, country_one, country_two)
    print(
        f'it seems like that in {country_one} you should mostly visit due to your love for {what_todo_in_country_one},\n '
        f'as for {country_two} it seems that you would enjoy because this country is great for {what_todo_in_country_two}')


if __name__ == "__main__":
    results = []
    test_countries = {'taiwan': 'culture', 'japan': 'culture', 'spain': 'culture'}

    # Loading the Data
    df = pd.read_excel('main.xlsx')
    # df = pd.read_excel('test.xlsx')

    # main(df)

    df = organize_data(df)
    df['Best'] = df['Best'].str.lower()
    df['Country'] = df['Country'].str.lower()

    # Global
    ALL_ATTRIBUTES = df['Best'].unique().tolist()
    ALL_COUNTRIES = df['Country'].unique().tolist()
    ALL_USERS = df['user_id'].unique().tolist()
    ATTRIBUTES_LEN = len(ALL_ATTRIBUTES)
    COUNTRIES_LEN = len(ALL_COUNTRIES)
    USERS_LEN = len(ALL_USERS)

    # Creating the Countries / Attributes matrix
    df_countries = pd.crosstab(df['Best'], df['Country'])

    # Normalize the Data
    for col in df_countries:
        df_countries[col] = df_countries[col] / df_countries[col].sum()

    # Creating the Users / Attributes matrix
    df_users = pd.crosstab(df['user_id'], df['Best'])
    # Normalize the Data
    df_users = df_users.div(df_users.sum(axis=1), axis=0)

    numpy_matrix = numpy.dot(df_users.values, df_countries.values)
    # Convert Back to DF
    matrix = normalizer(pd.DataFrame(numpy_matrix, columns=ALL_COUNTRIES, index=ALL_USERS))
    matrix.to_excel('matrix_factorization.xlsx')
