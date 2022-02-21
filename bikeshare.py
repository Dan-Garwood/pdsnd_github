import os
import time
import textwrap as tw
import pandas as pd
import numpy as np

# Set up the city data that can be imported.
CITY_DATA = {'chicago': 'chicago.csv',
             'new york city': 'new_york_city.csv',
             'washington': 'washington.csv'}

# Set a tuple of the months for which we have data, plus "all". We put 'all' in
# the zeroth position so the indices of each month match their values in
# pandas' dt.month values.
MONTH_DATA = ('all', 'january', 'february', 'march', 'april', 'may', 'june')

# Set a tuple of days of the week, plus "all". The order matches pandas'
# indexing for pandas' dt.dayofweek and equivalent datetime methods.
DAY_NAMES = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday',
             'saturday', 'sunday', 'all')

# Set a tuple of 12-hour-format strings for all 24 hours with indices matching
# their 24-hour-format equivalents (e.g., HOURS_TO_12[13] returns '1 pm').
HOURS_TO_12 = ('12 am', '1 am', '2 am', '3 am', '4 am', '5 am',
               '6 am', '7 am', '8 am', '9 am', '10 am', '11 am',
               '12 pm', '1 pm', '2 pm', '3 pm', '4 pm', '5 pm',
               '6 pm', '7 pm', '8 pm', '9 pm', '10 pm', '11 pm')

# Get info about the user's terminal environment.
term_width = min(70, os.get_terminal_size().columns)


def h_div(upper_nl=0, lower_nl=0, passthrough=False):
    """
    Prints a series of dashes to produce a horizontal divider.

    Args:
        upper_nl (int): The number of newline characters to print before the
            divider
        lower_nl (int): The number of newline characters to print after the
            divider
        passthrough (bool): If False, calls the print function; if True,
            passes the text without the print call so the text can be stored in
            a variable that can be printed later
    """
    div_str = ('\n' * upper_nl) + ('-' * term_width) + ('\n' * lower_nl)

    if passthrough is False:
        print(div_str)
    else:
        return div_str


def wrap(str):
    """
    Wraps a text string to the smaller of 70 characters or the user's terminal
    width.
    """
    wrapped = '\n'.join(tw.wrap(str, width=term_width,
                                replace_whitespace=False))
    return wrapped


def y_n_query(input_prompt='Yes or No? ',
              error_prompt='Input of Yes or No required. '):
    """
    Prompts the user to answer Yes or No (Y or N), case insensitve.

    Args:
        input_prompt (str): The prompt to the user
        error_prompt (str): The prompt if the input is not accepted

    Returns:
        (str): Returns 'yes' or 'no' depending on the user's input
    """
    y_or_n = input(input_prompt).lower()

    while y_or_n not in {'y', 'ye', 'ys', 'yes', 'n', 'no'}:
        y_or_n = input(error_prompt).lower()

    if y_or_n in {'n', 'no'}:
        return 'no'
    else:
        return 'yes'


# We will use the Levenshtein Ratio to correct for mistyped input. This code
# for the Levenshtein Distance and Ratio is adapated from
# https://www.datacamp.com/community/tutorials/fuzzy-string-python
# (Feb 15, 2022)
def lev_r_d(s, t, ratio_calc=False):
    """
    Calculates Levenshtein Distance or Ratio between two strings.

    Args:
        s (str): First of two strings to compare
        t (str): Second of two strings to compare
        ratio_calc (bool): If True, the function computes the Levenshtein
            Ratio (instead of Distance) of similarity between two strings

    Returns:
        lev_dist (int): If ratio_calc=False, returns the Levenshtein distance
            between the two strings
        lev_ratio (float): If ratio_calc=True, returns the Levenshtein ratio
            between the two strings
    """
    # Initialize matrix of zeros
    rows = len(s) + 1
    cols = len(t) + 1
    distance = np.zeros((rows, cols), dtype=int)

    # Populate matrix of zeros with the indeces of each character of both
    # strings
    for i in range(1, rows):
        for k in range(1, cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions
    # and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row - 1] == t[col - 1]:
                cost = 0  # If the characters are the same in the two strings
                          # in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python
                # Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just
                # distance, then the cost of a substitution is 1.
                if ratio_calc is True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row - 1][col] + 1,
                                     # Cost of deletions
                                     distance[row][col - 1] + 1,
                                     # Cost of insertions
                                     distance[row - 1][col - 1] + cost)
                                     # Cost of substitutions
    if ratio_calc is True:
        # Computation of the Levenshtein Distance Ratio
        lev_ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
        return lev_ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how
        # the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to
        # string b
        lev_dist = distance[row][col]
        return lev_dist


def partial_match(str_, list_, dedupe=True):
    """
    Finds the unique partial match of a string in a list.

    Args:
        str_ (str): The string to be compared to the elements of th list
        list_ (list): The list whose elements the string might match; elements
            must be strings
        dedupe (bool): If true, removes duplicates from the list before
            searching for a match; if false, the string matching a duplicate
            element will result in the function returning None

    Returns:
        match (str): The only element of the list that str partially matches;
            if there is not exactly one match, returns None
    """
    if dedupe is True:
        list_ = list(set(list_))  # Want a unique match, so can lose ordering

    matches = []
    # For each element of list_, check if str_ is a partial match.
    for item in list_:
        if str_ in item:
            matches.append(item)

    # If there is only one match, assign the match to the return variable.
    if len(matches) == 1:
        return matches[0]
    else:
        return None


def lev_best(str_, list_, ratio_calc=False):
    """
    Returns the element of a list that has the best Levenshtein result.

    Args:
        str_ (str): The string to be compared to the elements of the list
        list_ (list): The list whose elements might provide a good match to the
            string
        ratio_calc (bool): If false, uses a Levenshtein Distance test; if true,
            uses a Levenshtein Ratio test

    Returns:
        closest_item (str): The element from list_ that has the best Levenshtein
            test result to str_
        lev_result (int or float): The value of the best Levenshtein test
            result; int if a Distance test, float if a ratio test
    """
    dict = {}
    for item in list_:
        dict[item] = lev_r_d(str_, item, ratio_calc=ratio_calc)

    # Code to get min/max value from a dictonary's keys from
    # https://www.kite.com/python/answers/how-to-find-the-max-value-in-
    # a-dictionary-in-python (Feb 16, 2022)
    if ratio_calc is False:
        closest_item = min(dict, key=dict.get)
    elif ratio_calc is True:
        closest_item = max(dict, key=dict.get)
    lev_value = dict[closest_item]

    return closest_item, lev_value


def lev_confirm(closest_item, lev_value, min_ratio=.5, confirm_thresh=.75,
                reject_message='No match found.',
                input_prompt='Match: {}\nYes or No? ',
                error_prompt='Input of Yes or No required. '):
    """
    Processes the result of the lev_best function to confirm a good match.

    Args:
        closest_item (str): The closest_item string returned by lev_best
        lev_value (float): The lev_value float returned by lev_best
        min_ratio (float): The Levenshtein Ratio below which the lev_best
            result is automatically rejected
        confirm_thresh (float): The Levenshtein Ratio below which the user is
            asked to confirm the result
        reject_message (str): The message to print informing the user no good
            match was found
        input_prompt (str): The message to print requesting the user confirm
            whether the match is good
        error_prompt (str): The mssage to print if the user does not give a
            valid yes/no answer when confirming the match

    Returns:
        match (bool): Returns a boolean describing whether the match was
            accepted
    """
    # Code for checking whether the default value of an optional paramter is
    # set from https://stackoverflow.com/a/57628817 (Feb 17, 2022)
    # We use this to add the .format() to the input_message string,
    # since that can't be done in the default paramters
    if input_prompt is lev_confirm.__defaults__[3]:
        input_prompt = input_prompt.format(closest_item)

    if lev_value < min_ratio:
        print(reject_message)
        match = False

    elif lev_value < confirm_thresh:
        match = y_n_query(input_prompt=input_prompt, error_prompt=error_prompt)
        if match == 'yes':
            match = True
        else:
            match = False

    else:
        match = True

    return match


def get_filters():
    """
    Asks user to specify a city, month, and day to analyze.

    Returns:
        city (str): Name of the city to analyze
        month (str): Name of the month to filter by, or "all" to apply no
            month filter
        day (str): Name of the day of week to filter by, or "all" to apply no
            day filter
    """
    print(wrap('Hello! Let\'s explore some US bikeshare data!'))

    city = None
    while CITY_DATA.get(city) is None:
        city = input('\n' + wrap('Which city would you like to explore? '
                     'Please type Chicago, New York City, or Washington:')
                     + '\n> ').lower()

        # Check if the input uniquely matches part of a valid city. If yes,
        # reassign the city variable to the match so CITY_DATA.get(city) returns
        # a match.
        if CITY_DATA.get(city) is None:
            city_check = partial_match(city, list(CITY_DATA.keys()))
            if city_check is not None:
                city = city_check

                # If the city variable still does not hold a valid city,
                # perform a Levenshtein test to find the closest match among
                # valid cities.
            else:
                city_check, lev_value = lev_best(city, list(CITY_DATA.keys()),
                                                 ratio_calc=True)

                # Confirm the Levenshtein test returned a good match. First set
                # the messages for lev_confirm.
                reject_message = '\n' + wrap('Sorry, I don\'t have data for '
                                 'that city or I don\'t recognize the city '
                                 'name you entered. Please try again.') \
                                 + h_div(upper_nl=2, passthrough=True)
                input_prompt = '\n' + wrap('I think you want to explore data '
                               'for {}. Yes or No (Y/N)?'
                               .format(city_check.title())) + '\n> '
                error_prompt = '\n' + wrap('Sorry, I need Y or N to '
                               'continue.') + '\n> '
                match = lev_confirm(city_check, lev_value, min_ratio=.375,
                                    reject_message=reject_message,
                                    input_prompt=input_prompt,
                                    error_prompt=error_prompt)
                if match is True:
                    city = city_check

    # Get user input for month (all, january, february, ... , june)
    month = None
    while month not in MONTH_DATA:
        month = input('\n' + city.title() + '\n\n' + wrap('I have data for '
                      'January through June. Which month would you like data '
                      'for? You may also choose \"all\":' + '\n> ')).lower()

        # Check if the input uniquely matches part or all of a valid month. If
        # yes, reassign the month variable to the match.
        month_check = partial_match(month, list(MONTH_DATA))
        if month_check is not None:
            month = month_check

        # If the month is not in the MONTH_DATA tuple, test the user's input
        # for Levenshtein Ratio with the months in MONTH_DATA.
        else:
            month_check, lev_value = lev_best(month, list(MONTH_DATA),
                                              ratio_calc=True)

            # Confirm the Levenshtein test returned a good match. First set
            # the messages for lev_confirm.
            reject_message = '\n' + wrap('Sorry, I don\'t have data for '
                             'that month or I don\'t recognize the month '
                             'you entered. Please try again.') \
                             + h_div(upper_nl=2, passthrough=True)
            input_prompt = '\n' + wrap('I think you want to explore data '
                           'for {}. Yes or No (Y/N)?'
                           .format(month_check.title())) + '\n> '
            error_prompt = '\n' + wrap('Sorry, I need Y or N to '
                           'continue.') + '\n> '
            match = lev_confirm(month_check, lev_value,
                                reject_message=reject_message,
                                input_prompt=input_prompt,
                                error_prompt=error_prompt)
            if match is True:
                month = month_check

    # Get user input for day of week (all, monday, tuesday, ... sunday)
    day = None
    while day not in DAY_NAMES:
        day = input('\n' + city.title() + ' | ' + month.title() + '\n\n'
                    + wrap('Which day of the week would you like to analyze '
                    '(Monday, Tuesday, etc.)? You may also choose \"all\":')
                    + '\n> ').lower()

        # Check if the input uniquely matches part or all of a day name. If yes,
        # reassign the day variable to the match.
        day_check = partial_match(day, list(DAY_NAMES))
        if day_check is not None:
            day = day_check

        # If the day is not in the DAY_NAMES tuple, test the user's input for
        # Levenshtein Ratio with the days in DAY_NAMES.
        else:
            day_check, lev_value = lev_best(day, list(DAY_NAMES),
                                            ratio_calc=True)

            # Confirm the Levenshtein test returned a good match. First set
            # the messages for lev_confirm.
            reject_message = '\n' + wrap('Sorry, I don\'t recognize the day '
                             'you entered. Please try again.') \
                             + h_div(upper_nl=2, passthrough=True)
            input_prompt = '\n' + wrap('I think you want to explore data '
                            'for {}. Yes or No (Y/N)?'
                            .format(day_check.title())) + '\n> '
            error_prompt = '\n' + wrap('Sorry, I need Y or N to '
                            'continue.') + '\n> '
            match = lev_confirm(day_check, lev_value,
                                reject_message=reject_message,
                                input_prompt=input_prompt,
                                error_prompt=error_prompt)
            if match is True:
                day = day_check

    print('\n' + city.title() + ' | ' + month.title() + ' | ' + day.title())
    h_div(upper_nl=1, lower_nl=1)

    return city, month, day


def load_data(city, month, day):
    """
    Loads data for the specified city and filters by month and day if
    applicable.

    Args:
        city (str): Name of the city to analyze
        month (str): Name of the month to filter by, or "all" to apply no month
            filter
        day (str): Name of the day of week to filter by, or "all" to apply no
            day filter

    Returns:
        df: Pandas DataFrame containing city data filtered by month and day
    """
    df = pd.read_csv(CITY_DATA[city])
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df['End Time'] = pd.to_datetime(df['End Time'])
    df['Month'] = df['Start Time'].dt.month
    df['Day'] = df['Start Time'].dt.weekday
    df['Hour'] = df['Start Time'].dt.hour
    df['Trip'] = df['Start Station'] + ' to ' + df['End Station']

    if month != 'all':
        df = df[df['Month'] == MONTH_DATA.index(month)]
    if day != 'all':
        df = df[df['Day'] == DAY_NAMES.index(day)]

    return df


def time_stats(df, city, month, day):
    """Displays statistics on the most frequent times of travel."""

    print(wrap('Calculating The Most Frequent Times of Travel...') + '\n')
    start_time = time.time()

    print('\n' + wrap('{}\nFor trips on {}s in {}:'
          .format(city.title(), 'all day' if day == 'all' else day.title(),
                  'all months' if month == 'all' else month.title())) + '\n')

    # Display the most common month if user selected all months.
    if month == 'all':
        top_month = MONTH_DATA[df['Month'].value_counts(sort=True,
                ascending=False).index[0]]
        print(wrap('The most common month for trips is: {}'
              .format(top_month.title())))

    # Display the most common day of week if user selected all days.
    if day == 'all':
        top_day = DAY_NAMES[df['Day'].value_counts(sort=True,
                ascending=False).index[0]]
        print(wrap('The most common day for trips is: {}'
              .format(top_day.title())))

    # Display the most common start hour.
    top_hour = df['Hour'].value_counts(sort=True, ascending=False).index[0]
    print(wrap('The most common start hour is: {}'
          .format(HOURS_TO_12[top_hour])))

    print('\nThis analysis took %s seconds.' % (time.time() - start_time))


def station_stats(df, city, month, day):
    """Displays statistics on the most popular stations and trip."""

    print(wrap('Calculating The Most Popular Stations and Trip...') + '\n')
    start_time = time.time()

    print('\n' + wrap('{}\nFor trips on {}s in {}:'
          .format(city.title(), 'all day' if day == 'all' else day.title(),
                  'all months' if month == 'all' else month.title())) + '\n')

    # Display most commonly used start station.
    top_start = df['Start Station'].value_counts(sort=True,
                ascending=False).index[0]
    print(wrap('The most common start station is: {}'.format(top_start)))

    # Display most commonly used end station.
    top_end = df['End Station'].value_counts(sort=True,
              ascending=False).index[0]
    print(wrap('The most common end station is: {}'.format(top_end)))

    # Display most frequent combination of start station and end station trip.
    top_trip = df['Trip'].value_counts(sort=True, ascending=False).index[0]
    print(wrap('The most frequent trip is: {}'.format(top_trip)))

    if top_trip[:top_trip.find(' to ')] == top_trip[top_trip.find(' to ') + 4:]:
        # If the top trip is a round trip, also find the top one-way trip.
        top_oneway = df.loc[df['Start Station'] != df['End Station'],
            'Trip'].value_counts(sort=True, ascending=False).index[0]
        print(wrap('The most frequent one-way trip is: {}'.format(top_oneway)))

    print("\nThis analysis took %s seconds." % (time.time() - start_time))


def trip_duration_stats(df, city, month, day):
    """Displays statistics on the total and average trip duration."""

    print(wrap('Calculating Trip Duration...') + '\n')
    start_time = time.time()

    print('\n' + wrap('{}\nFor trips on {}s in {}:'
          .format(city.title(), 'all day' if day == 'all' else day.title(),
                  'all months' if month == 'all' else month.title())) + '\n')

    # Display total travel time
    duration_sum = df['Trip Duration'].sum()
    duration_sum = str(pd.Timedelta(duration_sum, unit='s'))
    duration_sum = duration_sum[:duration_sum.find('.')]
    print(wrap('The combined duration of all trips is: {} (hh:mm:ss)'
          .format(duration_sum)))

    # Display mean travel time
    duration_mean = df['Trip Duration'].mean()
    duration_mean = str(pd.Timedelta(duration_mean, unit='s'))
    duration_mean = duration_mean[duration_mean.find(':') + 1:duration_mean
                    .find('.')]
    print(wrap('The mean duration of trips is: {} (mm:ss)'
          .format(duration_mean)))

    print("\nThis analysis took %s seconds." % (time.time() - start_time))


def user_stats(df, city, month, day):
    """Displays statistics on bikeshare users."""

    print(wrap('Calculating User Stats...') + '\n')
    start_time = time.time()

    print('\n' + wrap('{}\nFor trips on {}s in {}:'
          .format(city.title(), 'all day' if day == 'all' else day.title(),
                  'all months' if month == 'all' else month.title())) + '\n')

    # Display counts of user types.
    user_types = df.groupby('User Type').size().sort_values(ascending=False)
    for type in user_types.index:
        print(wrap('{}s took {} trips'.format(type, user_types[type])
              .capitalize()))

    # Display counts of gender if it exists in the chosen city's data.
    try:
        genders = df.groupby('Gender').size().sort_values(ascending=False)
        print('')
        for gender in genders.index:
            print(wrap('{} users took {} trips'.format(gender, genders[gender])
                  .capitalize()))

    except KeyError:
        print(wrap('I don\'t have gender data for {}.'.format(city.title())))

    # Display earliest, most recent, and most common year of birth if birth
    # year exists in the chosen city's data.
    try:
        b_years = df.groupby('Birth Year').size().sort_index(ascending=True)
        b_years.index = b_years.index.astype('int32')  # Recast to int
        print('')
        print(wrap('The oldest users were born in: {}'
              .format(b_years.index[0])))
        print(wrap('The youngest users were born in: {}'
              .format(b_years.index[-1])))

        b_years = b_years.sort_values(ascending=False)
        print(wrap('The most common year of birth for users is: {}'
              .format(b_years.index[0])))

    except KeyError:
        print(wrap('I don\'t have birth year data for {}.'
              .format(city.title())))

    print("\nThis analysis took %s seconds." % (time.time() - start_time))


def wait_for_input(prompt='Press ENTER to continue...'):
    """Prompts the user to press ENTER to continue running the script."""
    h_div(upper_nl=1)
    input(prompt)
    h_div(lower_nl=1)


def view_raw(df, rows=5):
    """
    Prompts the user to view raw data in the DataFrame n rows at a time.

    Args:
        df: Pandas DataFrame
        rows (int): The number of rows to show at a time.
    """
    input_prompt = wrap('Would you like to view raw data? Yes or No (Y/N):') \
                   + '\n> '
    error_prompt = '\n' + wrap('Sorry, I need Y or N to continue.') + '\n> '

    see_raw = y_n_query(input_prompt=input_prompt, error_prompt=error_prompt)

    if see_raw == 'yes':
        # Remove the columns we created for filtering, since the filters have
        # already been applied.
        df = df.drop(['Month', 'Day', 'Hour', 'Trip'], axis=1)

        # Set options to show all columns. Code adapted from
        # https://thispointer.com/python-pandas-how-to-display-full-dataframe-
        # i-e-print-all-rows-columns-without-truncation/ (Feb 18, 2022)
        pd.set_option('display.max_rows', rows)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)

        # Set the index to track the number of rows the user wants to see, gets
        # the total available rows and sets the prompt to see more data. Code
        # for getting the number of rows in a DataFrame from
        # https://stackoverflow.com/q/15943769 (Feb 18, 2022)
        i = -rows
        row_count = len(df.index)
        input_prompt = '\n' + wrap('Would you like to see {} more rows? Yes '
                       'or No (Y/N):'.format(rows) + '\n> ')

    # Display the raw data n rows at a time, then prompt the user whether to
    # view the next n rows.
    while see_raw == 'yes' and (i + rows) < row_count:
        print('')
        i += rows
        print(df[:][i:i + rows])
        see_raw = y_n_query(input_prompt=input_prompt,
                            error_prompt=error_prompt)

    # Handle the end of the raw data, which may not be an intiger multiple of
    # the number of rows to display.
    if see_raw == 'yes' and (i + rows) >= row_count:
        rows_left = row_count - i
        print(df[:][i:i + rows_left])
        print('\n' + wrap('No rows left to display.'))
        return


def main():
    h_div(upper_nl=1, lower_nl=1)
    while True:
        city, month, day = get_filters()
        df = load_data(city, month, day)

        time_stats(df, city, month, day)
        wait_for_input(prompt=wrap('Press ENTER to view station stats...'))
        station_stats(df, city, month, day)
        wait_for_input(prompt=wrap('Press ENTER to view trip duration '
                       'stats...'))
        trip_duration_stats(df, city, month, day)
        wait_for_input(prompt=wrap('Press ENTER to view user stats...'))
        user_stats(df, city, month, day)
        h_div(upper_nl=1, lower_nl=1)
        view_raw(df)

        h_div(upper_nl=1, lower_nl=1)
        restart_prompt = wrap('Would you like to restart? Yes or No (Y/N):') \
                         + '\n> '
        error_prompt = '\n' + wrap('Sorry, I need Y or N to continue.') + '\n> '
        restart = y_n_query(input_prompt=restart_prompt,
                            error_prompt=error_prompt)

        if restart == 'yes':
            h_div(upper_nl=1)
            h_div(lower_nl=1)
        else:
            break


if __name__ == "__main__":
    main()
