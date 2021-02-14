import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# gameId, gameDuraton,
# blueWardPlaced, blueWardkills, blueTotalMinionKills, blueJungleMinionKills, blueTotalHeal,
#  redWardPlaced,  redWardkills,  redTotalMinionKills,  redJungleMinionKills,  redTotalHeal,
# win, FirstBlood, FirstTower, FirstBaron, FirstDragon


def full_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    color_map = plt.cm.get_cmap('rocket')
    reversed_color_map = color_map.reversed()
    df = df.replace('Red', 0)
    df = df.replace('Blue', 1)
    df = df.select_dtypes(np.number)
    matrix = np.triu(df.corr())
    sns.heatmap(df.corr(), annot=True, mask=matrix, cmap=reversed_color_map)
    plt.tight_layout()
    plt.show()


def outlier_box_plots(df):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)

    fig = sns.boxplot(y=df['gameDuraton'])
    fig.set_title('')
    fig.set_ylabel('gameDuraton')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['blueWardPlaced'])
    fig.set_title('')
    fig.set_ylabel('blueWardPlaced')

    plt.subplot(2, 2, 3)
    fig = sns.boxplot(y=df['blueWardkills'])
    fig.set_title('')
    fig.set_ylabel('blueWardkills')

    plt.subplot(2, 2, 4)
    fig = sns.boxplot(y=df['blueTotalMinionKills'])
    fig.set_title('')
    fig.set_ylabel('blueTotalMinionKills')

    plt.show()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = sns.boxplot(y=df['blueJungleMinionKills'])
    fig.set_title('')
    fig.set_ylabel('blueJungleMinionKills')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['blueTotalHeal'])
    fig.set_title('')
    fig.set_ylabel('blueTotalHeal')

    plt.subplot(2, 2, 3)
    fig = sns.boxplot(y=df['redWardPlaced'])
    fig.set_title('')
    fig.set_ylabel('redWardPlaced')

    plt.subplot(2, 2, 4)
    fig = sns.boxplot(y=df['redWardkills'])
    fig.set_title('')
    fig.set_ylabel('redWardkills')

    plt.show()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = sns.boxplot(y=df['redTotalMinionKills'])
    fig.set_title('')
    fig.set_ylabel('redTotalMinionKills')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['redJungleMinionKills'])
    fig.set_title('')
    fig.set_ylabel('redJungleMinionKills')

    plt.subplot(2, 2, 3)
    fig = sns.boxplot(y=df['redTotalHeal'])
    fig.set_title('')
    fig.set_ylabel('redTotalHeal')

    plt.show()


def outlier_dist_plot(df):
    sns.set_theme(style="whitegrid")

    sns.displot(df, x="gameDuraton", kde=True)
    plt.show()

    sns.displot(df, x="blueWardPlaced", kde=True)
    plt.show()

    sns.displot(df, x="blueWardkills", kde=True)
    plt.show()

    sns.displot(df, x="blueTotalMinionKills", kde=True)
    plt.show()

    sns.displot(df, x="blueJungleMinionKills", kde=True)
    plt.show()

    sns.displot(df, x="blueTotalHeal", kde=True)
    plt.show()

    sns.displot(df, x="redWardPlaced", kde=True)
    plt.show()

    sns.displot(df, x="redWardkills", kde=True)
    plt.show()

    sns.displot(df, x="redTotalMinionKills", kde=True)
    plt.show()

    sns.displot(df, x="redJungleMinionKills", kde=True)
    plt.show()

    sns.displot(df, x="redTotalHeal", kde=True)
    plt.show()


def first_blood_hist(df):
    df = df.replace('Red', 0)
    df = df.replace('Blue', 1)
    sns.kdeplot((df['FirstBlood'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstBlood'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstBlood                  blue won')
    plt.show()


def first_tower_hist(df):
    df = df.replace('Red', 0)
    df = df.replace('Blue', 1)
    sns.kdeplot((df['FirstTower'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstTower'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstTower                  blue won')
    plt.show()


def first_baron_hist(df):
    df = df.replace('Red', 0)
    df = df.replace('Blue', 1)
    sns.kdeplot((df['FirstBaron'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstBaron'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstBaron                  blue won')
    plt.show()


def first_dragon_hist(df):
    df = df.replace('Red', 0)
    df = df.replace('Blue', 1)
    sns.kdeplot((df['FirstDragon'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstDragon'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstDragon                  blue won')
    plt.show()


def combined_columns_bar_plots_num(df):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.countplot(x="wardPlaced", palette=['grey', 'green'], data=df)
    plt.xlabel('wardPlaced')

    plt.subplot(2, 2, 2)
    sns.countplot(x="wardKills", palette=['grey', 'green'], data=df)
    plt.xlabel('wardKills')

    plt.subplot(2, 2, 3)
    sns.countplot(x="totalMinionKills", palette=['grey', 'green'], data=df)
    plt.xlabel('totalMinionKills')

    plt.subplot(2, 2, 4)
    sns.countplot(x="jungleMinionKills", palette=['grey', 'green'], data=df)
    plt.xlabel('jungleMinionKills')

    plt.show()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.countplot(x="totalHeal", palette=['grey', 'green'], data=df)
    plt.xlabel('totalHeal')
    plt.show()


def combined_columns_bar_plots_cat(df):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    sns.countplot(x="FirstBlood", palette=['grey', 'green'], data=df)
    plt.xlabel('FirstBlood')

    plt.subplot(2, 2, 2)
    sns.countplot(x="FirstBaron", palette=['grey', 'green'], data=df)
    plt.xlabel('FirstBaron')

    plt.subplot(2, 2, 3)
    sns.countplot(x="FirstTower", palette=['grey', 'green'], data=df)
    plt.xlabel('FirstTower')

    plt.subplot(2, 2, 4)
    sns.countplot(x="FirstDragon", palette=['grey', 'green'], data=df)
    plt.xlabel('FirstDragon')

    plt.show()


def combined_columns_box_plots(df):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = sns.boxplot(y=df['winnerJungleMinionKills'], color='green')
    fig.set_title('')
    fig.set_ylabel('winnerJungleMinionKills')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['loserJungleMinionKills'], color='grey')
    fig.set_title('')
    fig.set_ylabel('loserJungleMinionKills')

    plt.subplot(2, 2, 3)
    fig = sns.boxplot(y=df['winnerWardPlaced'], color='green')
    fig.set_title('')
    fig.set_ylabel('winnerWardPlaced')

    plt.subplot(2, 2, 4)
    fig = sns.boxplot(y=df['loserWardPlaced'], color='grey')
    fig.set_title('')
    fig.set_ylabel('loserWardPlaced')

    plt.show()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = sns.boxplot(y=df['winnerWardKills'], color='green')
    fig.set_title('')
    fig.set_ylabel('winnerWardKills')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['loserWardKills'], color='grey')
    fig.set_title('')
    fig.set_ylabel('loserWardKills')

    plt.subplot(2, 2, 3)
    fig = sns.boxplot(y=df['winnerTotalMinionKills'], color='green')
    fig.set_title('')
    fig.set_ylabel('winnerTotalMinionKills')

    plt.subplot(2, 2, 4)
    fig = sns.boxplot(y=df['loserTotalMinionKills'], color='grey')
    fig.set_title('')
    fig.set_ylabel('loserTotalMinionKills')

    plt.show()

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    fig = sns.boxplot(y=df['winnerTotalHeal'], color='green')
    fig.set_title('')
    fig.set_ylabel('winnerTotalHeal')

    plt.subplot(2, 2, 2)
    fig = sns.boxplot(y=df['loserTotalHeal'], color='grey')
    fig.set_title('')
    fig.set_ylabel('loserTotalHeal')

    plt.show()


def duration_hist(df):
    sns.kdeplot((df['gameDuration'][df['win'] == 'Red']), color='red')
    sns.kdeplot((df['gameDuration'][df['win'] == 'Blue']), color='blue')
    plt.show()


def red_data_pair_plot(df):
    red_data = df.drop(['blueWardPlaced', 'blueWardKills', 'blueTotalMinionKills',
                        'blueJungleMinionKills', 'blueTotalHeal'], axis=1)
    sns.pairplot(red_data, hue='win', hue_order=['Blue', 'Red'], palette=['blue', 'red'])
    plt.show()


def blue_data_pair_plot(df):
    blue_data = df.drop(['redWardPlaced', 'redWardKills', 'redTotalMinionKills',
                         'redJungleMinionKills', 'redTotalHeal'], axis=1)
    sns.pairplot(blue_data, hue='win', hue_order=['Blue', 'Red'], palette=['blue', 'red'])
    plt.show()


def pair_plot(df, kind):
    sns.pairplot(data=df, hue='win', hue_order=['Blue', 'Red'], palette=['blue', 'red'], kind=kind,
                 y_vars=['redWardPlaced', 'redWardKills', 'redTotalMinionKills',
                         'redJungleMinionKills', 'redTotalHeal'],
                 x_vars=['blueWardPlaced', 'blueWardKills', 'blueTotalMinionKills',
                         'blueJungleMinionKills', 'blueTotalHeal'])
    plt.show()


def categorical_hist(df):
    # win, FirstBlood, FirstTower, FirstBaron, FirstDragon,
    # wardPlaced, wardKills, totalMinionKills, jungleMinionKills, totalHeal

    sns.kdeplot((df['FirstBlood'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstBlood'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstBlood                  blue won')
    plt.show()

    sns.kdeplot((df['FirstTower'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstTower'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstTower                  blue won')
    plt.show()

    sns.kdeplot((df['FirstBlood'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstBlood'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstBlood                  blue won')
    plt.show()

    sns.kdeplot((df['FirstDragon'][df['win'] == 0]), color='red')
    sns.kdeplot((df['FirstDragon'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  FirstDragon                  blue won')
    plt.show()

    sns.kdeplot((df['wardPlaced'][df['win'] == 0]), color='red')
    sns.kdeplot((df['wardPlaced'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  wardPlaced                  blue won')
    plt.show()

    sns.kdeplot((df['wardKills'][df['win'] == 0]), color='red')
    sns.kdeplot((df['wardKills'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  wardKills                  blue won')
    plt.show()

    sns.kdeplot((df['totalMinionKills'][df['win'] == 0]), color='red')
    sns.kdeplot((df['totalMinionKills'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  totalMinionKills                  blue won')
    plt.show()

    sns.kdeplot((df['jungleMinionKills'][df['win'] == 0]), color='red')
    sns.kdeplot((df['jungleMinionKills'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  jungleMinionKills                  blue won')
    plt.show()

    sns.kdeplot((df['totalHeal'][df['win'] == 0]), color='red')
    sns.kdeplot((df['totalHeal'][df['win'] == 1]), color='blue')
    plt.xlabel('red won                  totalHeal                  blue won')
    plt.show()


def six_zero_columns(df):
    six_zeros = df[(df["blueWardPlaced"] == 0) & (df["redWardPlaced"] == 0) &
                   (df["blueWardkills"] == 0) & (df["redWardkills"] == 0) &
                   (df["blueJungleMinionKills"] == 0) & (df["redJungleMinionKills"] == 0)]
    print(len(six_zeros))
    sns.displot(six_zeros, x="gameDuraton", kde=True)
    plt.show()


def clear_outliers(df):
    df.drop(df[(df["redWardPlaced"] > 190)].index, inplace=True)
    df.drop(df[(df["redWardkills"] > 100)].index, inplace=True)
    df.drop(df[(df["redTotalMinionKills"] > 1200)].index, inplace=True)
    df.drop(df[(df["redJungleMinionKills"] > 350)].index, inplace=True)
    df.drop(df[(df["redTotalHeal"] > 150000)].index, inplace=True)
    df.drop(df[(df["blueWardPlaced"] > 200)].index, inplace=True)
    df.drop(df[(df["blueWardkills"] > 100)].index, inplace=True)
    df.drop(df[(df["blueTotalMinionKills"] > 1190)].index, inplace=True)
    df.drop(df[(df["blueTotalHeal"] > 130000)].index, inplace=True)


def clear_six_zero_rows(df):
    df.drop(df[(df["blueWardPlaced"] == 0) & (df["redWardPlaced"] == 0) &
               (df["blueWardkills"] == 0) & (df["redWardkills"] == 0) &
               (df["blueJungleMinionKills"] == 0) & (df["redJungleMinionKills"] == 0)].index, inplace=True)


def replace_missing_values(df):
    df['blueWardPlaced'].fillna(round((df['blueWardPlaced'].mean()), 2), inplace=True)
    df['blueWardkills'].fillna(round((df['blueWardkills'].mean()), 2), inplace=True)
    df['blueTotalMinionKills'].fillna(round((df['blueTotalMinionKills'].mean()), 2), inplace=True)
    df['blueJungleMinionKills'].fillna(round((df['blueJungleMinionKills'].mean()), 2), inplace=True)
    df['blueTotalHeal'].fillna(round((df['blueTotalHeal'].mean()), 2), inplace=True)

    df['redWardPlaced'].fillna(round((df['redWardPlaced'].mean()), 2), inplace=True)
    df['redWardkills'].fillna(round((df['redWardkills'].mean()), 2), inplace=True)
    df['redTotalMinionKills'].fillna(round((df['redTotalMinionKills'].mean()), 2), inplace=True)
    df['redJungleMinionKills'].fillna(round((df['redJungleMinionKills'].mean()), 2), inplace=True)
    df['redTotalHeal'].fillna(round((df['redTotalHeal'].mean()), 2), inplace=True)

    df['FirstBlood'].fillna((df['FirstBlood'].mode()[0]), inplace=True)
    df['FirstTower'].fillna((df['FirstTower'].mode()[0]), inplace=True)
    df['FirstBaron'].fillna((df['FirstBaron'].mode()[0]), inplace=True)
    df['FirstDragon'].fillna((df['FirstDragon'].mode()[0]), inplace=True)


def combine_numerical_columns(df):
    df['winnerWardPlaced'] = df.apply(
        lambda row: compare_winning_team_num(row['redWardPlaced'], row['blueWardPlaced'], row['win']), axis=1)
    df['winnerWardKills'] = df.apply(
        lambda row: compare_winning_team_num(row['redWardKills'], row['blueWardKills'], row['win']), axis=1)
    df['winnerTotalMinionKills'] = df.apply(
        lambda row: compare_winning_team_num(row['redTotalMinionKills'], row['blueTotalMinionKills'], row['win']),
        axis=1)
    df['winnerJungleMinionKills'] = df.apply(
        lambda row: compare_winning_team_num(row['redJungleMinionKills'], row['blueJungleMinionKills'], row['win']),
        axis=1)
    df['winnerTotalHeal'] = df.apply(
        lambda row: compare_winning_team_num(row['redTotalHeal'], row['blueTotalHeal'], row['win']), axis=1)

    df['loserWardPlaced'] = df.apply(
        lambda row: compare_losing_team_num(row['redWardPlaced'], row['blueWardPlaced'], row['win']), axis=1)
    df['loserWardKills'] = df.apply(
        lambda row: compare_losing_team_num(row['redWardKills'], row['blueWardKills'], row['win']), axis=1)
    df['loserTotalMinionKills'] = df.apply(
        lambda row: compare_losing_team_num(row['redTotalMinionKills'], row['blueTotalMinionKills'], row['win']),
        axis=1)
    df['loserJungleMinionKills'] = df.apply(
        lambda row: compare_losing_team_num(row['redJungleMinionKills'], row['blueJungleMinionKills'], row['win']),
        axis=1)
    df['loserTotalHeal'] = df.apply(
        lambda row: compare_losing_team_num(row['redTotalHeal'], row['blueTotalHeal'], row['win']), axis=1)

    df.drop(['blueWardPlaced', 'redWardPlaced', 'blueWardKills', 'redWardKills',
             'blueTotalMinionKills', 'redTotalMinionKills', 'blueJungleMinionKills', 'redJungleMinionKills',
             'blueTotalHeal', 'redTotalHeal'], axis=1, inplace=True)

    return df


def compare_winning_team_num(red_val, blue_val, winner):
    if winner == 'Blue':
        return blue_val
    else:
        return red_val


def compare_losing_team_num(red_val, blue_val, winner):
    if winner == 'Blue':
        return red_val
    else:
        return blue_val


def combine_categorical_columns(df):
    df['FirstBlood'] = df.apply(lambda row: compare_team_cat(row['FirstBlood'], row['win']), axis=1)
    df['FirstTower'] = df.apply(lambda row: compare_team_cat(row['FirstTower'], row['win']), axis=1)
    df['FirstBaron'] = df.apply(lambda row: compare_team_cat(row['FirstBaron'], row['win']), axis=1)
    df['FirstDragon'] = df.apply(lambda row: compare_team_cat(row['FirstDragon'], row['win']), axis=1)

    return df


def compare_team_cat(val, winner):
    if winner == val:
        return 1
    else:
        return 0


def stat_diff_winner_loser_num(df):
    win_val = df['winnerWardPlaced'].mean()
    lose_val = df['loserWardPlaced'].mean()
    print('winnerWardPlaced:        ', round(win_val, 2))
    print('loserWardPlaced:         ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['winnerWardKills'].mean()
    lose_val = df['loserWardKills'].mean()
    print('winnerWardKills:         ', round(win_val, 2))
    print('loserWardKills:          ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['winnerTotalMinionKills'].mean()
    lose_val = df['loserTotalMinionKills'].mean()
    print('winnerTotalMinionKills:  ', round(win_val, 2))
    print('loserTotalMinionKills:   ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['winnerJungleMinionKills'].mean()
    lose_val = df['loserJungleMinionKills'].mean()
    print('winnerJungleMinionKills: ', round(win_val, 2))
    print('loserJungleMinionKills:  ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['winnerTotalHeal'].mean()
    lose_val = df['loserTotalHeal'].mean()
    print('winnerTotalHeal:         ', round(win_val, 2))
    print('loserTotalHeal:          ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)


def stat_diff_winner_loser_cat(df):
    win_val = df['FirstBlood'].mean()
    lose_val = 1 - win_val
    print('winnerFirstBlood:        ', round(win_val, 2))
    print('loserFirstBlood:         ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['FirstTower'].mean()
    lose_val = 1 - win_val
    print('winnerFirstTower:        ', round(win_val, 2))
    print('loserFirstTower:         ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['FirstBaron'].mean()
    lose_val = 1 - win_val
    print('winnerFirstBaron:        ', round(win_val, 2))
    print('loserFirstBaron:         ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['FirstDragon'].mean()
    lose_val = 1 - win_val
    print('winnerFirstDragon:       ', round(win_val, 2))
    print('loserFirstDragon:        ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)


def stat_diff_winner_loser_num_binary(df):
    win_val = df['wardPlaced'].mean()
    lose_val = 1 - win_val
    print('winnerWardPlaced:        ', round(win_val, 2))
    print('loserWardPlaced:         ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['wardKills'].mean()
    lose_val = 1 - win_val
    print('winnerWardKills:         ', round(win_val, 2))
    print('loserWardKills:          ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['totalMinionKills'].mean()
    lose_val = 1 - win_val
    print('winnerTotalMinionKills:  ', round(win_val, 2))
    print('loserTotalMinionKills:   ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['jungleMinionKills'].mean()
    lose_val = 1 - win_val
    print('winnerJungleMinionKills: ', round(win_val, 2))
    print('loserJungleMinionKills:  ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)

    win_val = df['totalHeal'].mean()
    lose_val = 1 - win_val
    print('winnerTotalHeal:         ', round(win_val, 2))
    print('loserTotalHeal:          ', round(lose_val, 2))
    calculate_percent(win_val, lose_val)


def calculate_percent(win_val, lose_val):
    increase_percent = ((win_val - lose_val) / lose_val) * 100
    print('increase percent:        ', round(increase_percent, 2), '%\n')


def combine_numerical_columns_binary(df):
    df['wardPlaced'] = df.apply(
        lambda row: compare_team_cat(winner_team(row['redWardPlaced'], row['blueWardPlaced']), row['win']), axis=1)
    df['wardKills'] = df.apply(
        lambda row: compare_team_cat(winner_team(row['redWardKills'], row['blueWardKills']), row['win']), axis=1)
    df['totalMinionKills'] = df.apply(
        lambda row: compare_team_cat(winner_team(row['redTotalMinionKills'], row['blueTotalMinionKills']), row['win']),
        axis=1)
    df['jungleMinionKills'] = df.apply(
        lambda row: compare_team_cat(winner_team(row['redJungleMinionKills'], row['blueJungleMinionKills']),
                                     row['win']), axis=1)
    df['totalHeal'] = df.apply(
        lambda row: compare_team_cat(winner_team(row['redTotalHeal'], row['blueTotalHeal']), row['win']), axis=1)

    df.drop(['blueWardPlaced', 'redWardPlaced', 'blueWardKills', 'redWardKills',
             'blueTotalMinionKills', 'redTotalMinionKills', 'blueJungleMinionKills', 'redJungleMinionKills',
             'blueTotalHeal', 'redTotalHeal'], axis=1, inplace=True)

    return df


def winner_team(red_val, blue_val):
    if red_val > blue_val:
        return 'Red'
    else:
        return 'Blue'


def initial_data_analysis():
    df = pd.read_csv("lol5.csv")
    df.drop(['gameId'], axis=1, inplace=True)

    print(df.isnull().sum())
    replace_missing_values(df)

    first_baron_hist(df)
    first_tower_hist(df)
    first_blood_hist(df)
    first_dragon_hist(df)

    print(df.describe())

    outlier_box_plots(df)
    outlier_dist_plot(df)

    six_zero_columns(df)

    clear_six_zero_rows(df)
    clear_outliers(df)

    outlier_box_plots(df)
    outlier_dist_plot(df)

    df.rename(columns={
        'gameDuraton': 'gameDuration',
        'blueWardkills': 'blueWardKills',
        'redWardkills': 'redWardKills'
    }, inplace=True)

    df.to_csv('clean_lol.csv', index=False)


def exploratory_data_analysis():
    df = pd.read_csv("clean_lol.csv")
    full_correlation_heatmap(df)
    duration_hist(df)

    red_data_pair_plot(df)
    blue_data_pair_plot(df)
    pair_plot(df, 'scatter')
    pair_plot(df, 'kde')

    df = combine_numerical_columns(df)
    combined_columns_box_plots(df)
    stat_diff_winner_loser_num(df)
    df.to_csv('win_vs_lose.csv', index=False)

    df = pd.read_csv("clean_lol.csv")

    df = combine_categorical_columns(df)
    stat_diff_winner_loser_cat(df)
    combined_columns_bar_plots_cat(df)

    df = combine_numerical_columns_binary(df)
    stat_diff_winner_loser_num_binary(df)
    combined_columns_bar_plots_num(df)
    df.to_csv('binary_lol.csv', index=False)


if __name__ == '__main__':
    print('1: ')
    # initial_data_analysis()

    print('2: ')
    # exploratory_data_analysis()

    # data = pd.read_csv("clean_lol.csv")

    # correlation_heatmap_numerical()
    # print(data.head())
    # correlation_heatmap_numerical()

    # data = combine_columns()

    # data = data.drop(['FirstBlood', 'FirstTower', 'FirstBaron', 'FirstDragon'])
    # data.to_csv('win_vs_lose_num.csv', index=False)
    #
    # data = pd.read_csv("clean_lol.csv")
    # data = data[['win', 'FirstBlood', 'FirstTower', 'FirstBaron', 'FirstDragon']]
    # data = combine_categorical_columns()
    # data.to_csv('win_vs_lose_cat.csv', index=False)
    # #
    # data = pd.read_csv("lol5.csv")
    # data.dropna(inplace=True)
    # data = data[['win', 'FirstBlood', 'FirstTower', 'FirstBaron', 'FirstDragon']]
    # data = combine_categorical_columns()
    # data.to_csv('win_vs_lose_cat_origin.csv', index=False)
    # print('origin:')
    # data = pd.read_csv("win_vs_lose_cat_origin.csv")
    # # stat_diff_winner_loser_cat()
    # combined_columns_bar_plots()
    # # sns.countplot(x="winnerFirstBlood", palette=['grey', 'green'], data=data)
    # # plt.xlabel('FirstBlood')
    # # plt.show()
    #
    # print('---------------------')
    # print('clean:')
    # data = pd.read_csv("clean_lol.csv")
    # sns.pairplot(data, corner=True)
    # plt.show()

    # data = combine_categorical_columns()
    # print(data.head())

    # data.drop(['gameDuration', 'FirstBlood', 'FirstTower', 'FirstBaron', 'FirstDragon'], inplace=True, axis=1)

    # combine_numerical_columns_binary()

    # print(data.head())
    # pair_plot()


    # data.replace('Red', 0, inplace=True)
    # data.replace('Blue', 1, inplace=True)
    # full_cat = combine_columns()
    # full_num = data.drop(['win', 'FirstBlood', 'FirstTower', 'FirstBaron', 'FirstDragon'])
    # print(full_cat.head())
    # full_cat.to_csv('full_cat.csv', index=False)
