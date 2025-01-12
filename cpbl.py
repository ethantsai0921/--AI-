import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import torch.optim as optim
import pickle
from tqdm import tqdm


pd.set_option("display.max_columns", None)
pd.set_option("display.max_seq_items", None)

mlb_battingstats = pd.read_csv(
    filepath_or_buffer=r"C:\Users\user\Downloads\ASIA_batting_stats_precision_use.csv",
    encoding="utf_8",
    sep=",",
)
cpbl_csv = pd.read_csv(
    filepath_or_buffer=r"C:\Users\user\Downloads\CPBL_batting_stats_precision_use.csv",
    encoding="utf_8",
    sep=",",
)


# cpbl_csv = cpbl_csv.dropna()


# ------------------------------------------------------------------------------------------------------------------------------


def h(playername):
    batting_info = mlb_battingstats[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]

    mlb_batter_x = batting_info[
        [
            "g",
            "pa",
            "ab",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    mlb_batter_y = batting_info[["h"]]
    mlb_batter_x_train, mlb_batter_x_test, mlb_batter_y_train, mlb_batter_y_test = (
        train_test_split(mlb_batter_x, mlb_batter_y, train_size=0.8, test_size=0.2)
    )

    baseball_batter_model = LinearRegression()
    baseball_batter_model.fit(mlb_batter_x_train, mlb_batter_y_train)

    mlb_batter_y_predict = baseball_batter_model.predict(mlb_batter_x_test)

    plt.scatter(mlb_batter_y_test, mlb_batter_y_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test))

    mlb_batter_weights = baseball_batter_model.coef_

    cpbl_batter_x = cpbl_csv[
        [
            "g",
            "pa",
            "ab",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    cpbl_batter_y = cpbl_csv[["h"]]

    cpbl_batter_predict = baseball_batter_model.predict(cpbl_batter_x)
    plt.scatter(cpbl_batter_y, cpbl_batter_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted CPBL")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y))

    cpbl_data = cpbl_csv
    x_data = torch.FloatTensor(np.array(cpbl_batter_x))
    y_data = torch.FloatTensor(np.array(cpbl_batter_y))

    epochs = 30000

    targets_data = mlb_batter_weights.T
    targets_df = pd.DataFrame(data=targets_data)
    targets_df.columns = ["targets"]
    W = torch.Tensor(np.array(mlb_batter_weights).T)
    W = W.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-6)

    for epoch in tqdm(range(epochs + 1), desc="Training", unit="epoch"):
        hypothesis = x_data.matmul(W) + b

        cost = torch.mean((hypothesis - y_data) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        # print(
        #     "Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} ".format(
        #         epoch, epochs, hypothesis.squeeze().detach(), cost.item()
        #     )
        # )
        if epoch == epochs:
            hypo = hypothesis.detach().numpy()

    hypo = np.ravel(hypo, order="C")
    total = 0
    for s in tqdm(range(len(hypo)), desc="Calculating", unit="player"):
        total += hypo[s]

    cpbl_batter_y = np.array(cpbl_batter_y)
    cpbl_batter_y = np.ravel(cpbl_batter_y, order="C")
    total2 = 0
    for p in range(len(cpbl_batter_y)):
        total2 += cpbl_batter_y[p]

    cpbl_score = 1 - (abs(total - total2) / (total2))

    print(
        "MLB模型評估MLB選手時的準確度 =                {:.6f}".format(
            baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test)
        )
    )
    print(
        "MLB模型不調整評估CPBL選手時的準確度 =   {:.6f}".format(
            baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y)
        )
    )
    print(
        "MLB模型調整後評估CPBL選手時的準確度 =        {:.6f}".format(
            np.mean(cpbl_score)
        )
    )
    print("計算的最終權重 = ")
    print(W)

    cpbl_predictions = pd.DataFrame(
        {"Actual H": cpbl_batter_y.flatten(), "Predicted H": hypo}
    )

    print(cpbl_predictions)

    if playername in cpbl_csv["name_zh"].values:
        b_ex = cpbl_csv[
            [
                "name_en",
                "name_zh",
                "team",
                "league",
                "year",
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        print(b_ex[b_ex["name_zh"] == playername])
        print("\n\n")
        b_ex2 = b_ex[b_ex.name_zh == playername]
        input_b = b_ex2[
            [
                "g",
                "pa",
                "ab",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        output_b = baseball_batter_model.predict(input_b)
        print(output_b)

        player_data = cpbl_csv[cpbl_csv["name_zh"] == playername]
        player_data = player_data.sort_values(by="year")
        actual_h = player_data["h"]
        predicted_h = output_b
        plt.plot(player_data['year'], actual_h, label='Actual', marker='o')
        plt.plot(player_data['year'], predicted_h, label='Predicted', marker='x')
        plt.xlabel('Year')
        plt.ylabel('H (Hits)')
        plt.title('Actual vs Predicted Hits Over Time')
        plt.legend()
        plt.show()
    
        x = player_data["year"].values.reshape(-1, 1)
        y = player_data["h"].values

        model = LinearRegression()
        model.fit(x, y)
   
        future_years = np.arange(x[-1] + 1, x[-1] + 6).reshape(-1, 1)  
        future_predictions = model.predict(future_years)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="Actual", marker="o")
        plt.plot(future_years, future_predictions, label="Predicted Future", marker="x", linestyle="--")
        plt.xlabel("Year")
        plt.ylabel("Hits (H)")  
        plt.title(f"Time Series Prediction for {playername}")
        plt.legend()
        plt.grid()
        plt.show()
    

# ----------------------------------------------------------------------------------------------------------------------------------------------
"G", "AB", "R", "RBI", "BB", "SO", "HBP", "SF", "SH", "SB", "CS", "OBP"


def avg(playername):
    batting_info = mlb_battingstats[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]

    mlb_batter_x = batting_info[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    mlb_batter_y = batting_info[["avg"]]
    mlb_batter_x_train, mlb_batter_x_test, mlb_batter_y_train, mlb_batter_y_test = (
        train_test_split(mlb_batter_x, mlb_batter_y, train_size=0.8, test_size=0.2)
    )

    baseball_batter_model = LinearRegression()
    baseball_batter_model.fit(mlb_batter_x_train, mlb_batter_y_train)

    mlb_batter_y_predict = baseball_batter_model.predict(mlb_batter_x_test)

    plt.scatter(mlb_batter_y_test, mlb_batter_y_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    # print(baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test))

    mlb_batter_weights = baseball_batter_model.coef_

    cpbl_batter_x = cpbl_csv[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    cpbl_batter_y = cpbl_csv[["avg"]]

    cpbl_batter_predict = baseball_batter_model.predict(cpbl_batter_x)
    plt.scatter(cpbl_batter_y, cpbl_batter_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted CPBL")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    # print(baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y))

    cpbl_data = cpbl_csv
    x_data = torch.FloatTensor(np.array(cpbl_batter_x))
    y_data = torch.FloatTensor(np.array(cpbl_batter_y))

    epochs = 30000

    targets_data = mlb_batter_weights.T
    targets_df = pd.DataFrame(data=targets_data)
    targets_df.columns = ["targets"]
    W = torch.Tensor(np.array(mlb_batter_weights).T)
    W = W.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-6)

    for epoch in tqdm(range(epochs + 1), desc="Training", unit="epoch"):
        hypothesis = x_data.matmul(W) + b

        cost = torch.mean((hypothesis - y_data) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        # print(
        #     "Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} ".format(
        #         epoch, epochs, hypothesis.squeeze().detach(), cost.item()
        #     )
        # )
        if epoch == epochs:
            hypo = hypothesis.detach().numpy()

    hypo = np.ravel(hypo, order="C")
    total = 0
    for s in tqdm(range(len(hypo)), desc="Calculating", unit="player"):
        total += hypo[s]

    cpbl_batter_y = np.array(cpbl_batter_y)
    cpbl_batter_y = np.ravel(cpbl_batter_y, order="C")
    total2 = 0
    for p in range(len(cpbl_batter_y)):
        total2 += cpbl_batter_y[p]

    cpbl_score = 1 - (abs(total - total2) / (total2))

    print(
        "MLB模型評估MLB選手時的準確度 =                {:.6f}".format(
            baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test)
        )
    )
    print(
        "MLB模型不調整評估CPBL選手時的準確度 =   {:.6f}".format(
            baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y)
        )
    )
    print(
        "MLB模型調整後評估CPBL選手時的準確度 =        {:.6f}".format(
            np.mean(cpbl_score)
        )
    )
    print("計算的最終權重 = ")
    print(W)

    cpbl_predictions = pd.DataFrame(
        {"Actual AVG": cpbl_batter_y.flatten(), "Predicted AVG": hypo}
    )

    print(cpbl_predictions)

    if playername in cpbl_csv["name_zh"].values:
        b_ex = cpbl_csv[
            [
                "name_en",
                "name_zh",
                "team",
                "league",
                "year",
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        print(b_ex[b_ex["name_zh"] == playername])
        print("\n\n")
        b_ex2 = b_ex[b_ex.name_zh == playername]
        input_b = b_ex2[
            [
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        output_b = baseball_batter_model.predict(input_b)
        print(output_b)

        player_data = cpbl_csv[cpbl_csv["name_zh"] == playername]
        player_data = player_data.sort_values(by="year")
        actual_avg = player_data["avg"]
        predicted_avg = output_b
        plt.plot(player_data['year'], actual_avg, label='Actual', marker='o')
        plt.plot(player_data['year'], predicted_avg, label='Predicted', marker='x')
        plt.xlabel('Year')
        plt.ylabel('AVG (batting)')
        plt.title('Actual vs Predicted AVG (batting) Over Time')
        plt.legend()
        plt.show()

        x = player_data["year"].values.reshape(-1, 1)
        y = player_data["avg"].values

        model = baseball_batter_model
        model.fit(x, y)
        future_years = np.arange(x[-1] + 1, x[-1] + 6).reshape(-1, 1)
        future_predictions = model.predict(future_years)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="Actual", marker="o")
        plt.plot(future_years, future_predictions, label="Predicted Future", marker="x", linestyle="--")
        plt.xlabel("Year")
        plt.ylabel("AVG (batting)")
        plt.title(f"Time Series Prediction for {playername}")
        plt.legend()
        plt.grid()
        plt.show()

        
# --------------------------------------------------------------------------------------------------------------------------------------------


def bb(playername):
    batting_info = mlb_battingstats[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]

    mlb_batter_x = batting_info[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    mlb_batter_y = batting_info[["bb"]]
    mlb_batter_x_train, mlb_batter_x_test, mlb_batter_y_train, mlb_batter_y_test = (
        train_test_split(mlb_batter_x, mlb_batter_y, train_size=0.8, test_size=0.2)
    )

    baseball_batter_model = LinearRegression()
    baseball_batter_model.fit(mlb_batter_x_train, mlb_batter_y_train)

    mlb_batter_y_predict = baseball_batter_model.predict(mlb_batter_x_test)

    plt.scatter(mlb_batter_y_test, mlb_batter_y_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test))

    mlb_batter_weights = baseball_batter_model.coef_

    cpbl_batter_x = cpbl_csv[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    cpbl_batter_y = cpbl_csv[["bb"]]

    cpbl_batter_predict = baseball_batter_model.predict(cpbl_batter_x)
    plt.scatter(cpbl_batter_y, cpbl_batter_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted CPBL")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y))

    cpbl_data = cpbl_csv
    x_data = torch.FloatTensor(np.array(cpbl_batter_x))
    y_data = torch.FloatTensor(np.array(cpbl_batter_y))

    epochs = 30000

    targets_data = mlb_batter_weights.T
    targets_df = pd.DataFrame(data=targets_data)
    targets_df.columns = ["targets"]
    W = torch.Tensor(np.array(mlb_batter_weights).T)
    W = W.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-6)

    for epoch in tqdm(range(epochs + 1), desc="Training", unit="epoch"):
        hypothesis = x_data.matmul(W) + b

        cost = torch.mean((hypothesis - y_data) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        # print(
        #     "Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} ".format(
        #         epoch, epochs, hypothesis.squeeze().detach(), cost.item()
        #     )
        # )
        if epoch == epochs:
            hypo = hypothesis.detach().numpy()

    hypo = np.ravel(hypo, order="C")
    total = 0
    for s in tqdm(range(len(hypo)), desc="Calculating", unit="player"):
        total += hypo[s]

    cpbl_batter_y = np.array(cpbl_batter_y)
    cpbl_batter_y = np.ravel(cpbl_batter_y, order="C")
    total2 = 0
    for p in range(len(cpbl_batter_y)):
        total2 += cpbl_batter_y[p]

    cpbl_score = 1 - (abs(total - total2) / (total2))

    print(
        "MLB模型評估MLB選手時的準確度 =                {:.6f}".format(
            baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test)
        )
    )
    print(
        "MLB模型不調整評估CPBL選手時的準確度 =   {:.6f}".format(
            baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y)
        )
    )
    print(
        "MLB模型調整後評估CPBL選手時的準確度 =        {:.6f}".format(
            np.mean(cpbl_score)
        )
    )
    print("計算的最終權重 = ")
    print(W)

    cpbl_predictions = pd.DataFrame(
        {"Actual BB": cpbl_batter_y.flatten(), "Predicted BB": hypo}
    )

    print(cpbl_predictions)

    if playername in cpbl_csv["name_zh"].values:
        b_ex = cpbl_csv[
            [
                "name_en",
                "name_zh",
                "team",
                "league",
                "year",
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        print(b_ex[b_ex["name_zh"] == playername])
        print("\n\n")
        b_ex2 = b_ex[b_ex.name_zh == playername]
        input_b = b_ex2[
            [
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        output_b = baseball_batter_model.predict(input_b)
        print(output_b)


# ---------------------------------------------------------------------------------------------------------------------------------------------------


def g(playername):
    batting_info = mlb_battingstats[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]

    mlb_batter_x = batting_info[
        [
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    mlb_batter_y = batting_info[["g"]]
    mlb_batter_x_train, mlb_batter_x_test, mlb_batter_y_train, mlb_batter_y_test = (
        train_test_split(mlb_batter_x, mlb_batter_y, train_size=0.8, test_size=0.2)
    )

    baseball_batter_model = LinearRegression()
    baseball_batter_model.fit(mlb_batter_x_train, mlb_batter_y_train)

    mlb_batter_y_predict = baseball_batter_model.predict(mlb_batter_x_test)

    plt.scatter(mlb_batter_y_test, mlb_batter_y_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test))

    mlb_batter_weights = baseball_batter_model.coef_

    cpbl_batter_x = cpbl_csv[
        [
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    cpbl_batter_y = cpbl_csv[["g"]]

    cpbl_batter_predict = baseball_batter_model.predict(cpbl_batter_x)
    plt.scatter(cpbl_batter_y, cpbl_batter_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted CPBL")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y))

    cpbl_data = cpbl_csv
    x_data = torch.FloatTensor(np.array(cpbl_batter_x))
    y_data = torch.FloatTensor(np.array(cpbl_batter_y))

    epochs = 30000

    targets_data = mlb_batter_weights.T
    targets_df = pd.DataFrame(data=targets_data)
    targets_df.columns = ["targets"]
    W = torch.Tensor(np.array(mlb_batter_weights).T)
    W = W.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-6)

    for epoch in range(epochs + 1):
        hypothesis = x_data.matmul(W) + b

        cost = torch.mean((hypothesis - y_data) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(
            "Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} ".format(
                epoch, epochs, hypothesis.squeeze().detach(), cost.item()
            )
        )
        if epoch == epochs:
            hypo = hypothesis.detach().numpy()

    hypo = np.ravel(hypo, order="C")
    total = 0
    for s in range(len(hypo)):
        total += hypo[s]

    cpbl_batter_y = np.array(cpbl_batter_y)
    cpbl_batter_y = np.ravel(cpbl_batter_y, order="C")
    total2 = 0
    for p in range(len(cpbl_batter_y)):
        total2 += cpbl_batter_y[p]

    cpbl_score = 1 - (abs(total - total2) / (total2))

    print(
        "MLB模型評估MLB選手時的準確度 =                {:.6f}".format(
            baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test)
        )
    )
    print(
        "MLB模型不調整評估CPBL選手時的準確度 =   {:.6f}".format(
            baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y)
        )
    )
    print(
        "MLB模型調整後評估CPBL選手時的準確度 =        {:.6f}".format(
            np.mean(cpbl_score)
        )
    )
    print("計算的最終權重 = ")
    print(W)

    cpbl_predictions = pd.DataFrame(
        {"Actual g": cpbl_batter_y.flatten(), "Predicted g": hypo}
    )

    print(cpbl_predictions)

    if playername in cpbl_csv["player"].values:
        b_ex = cpbl_csv[
            [
                "player",
                "team",
                "league",
                "year",
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        print(b_ex[b_ex["player"] == playername])
        print("\n\n")
        b_ex2 = b_ex[b_ex.player == playername]
        input_b = b_ex2[
            [
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        output_b = baseball_batter_model.predict(input_b)
        print(output_b)


# -------------------------------------------------------------------------------------------------------------------------------------------------------


def hr(playername):
    batting_info = mlb_battingstats[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]

    mlb_batter_x = batting_info[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    mlb_batter_y = batting_info[["hr"]]
    mlb_batter_x_train, mlb_batter_x_test, mlb_batter_y_train, mlb_batter_y_test = (
        train_test_split(mlb_batter_x, mlb_batter_y, train_size=0.8, test_size=0.2)
    )

    baseball_batter_model = LinearRegression()
    baseball_batter_model.fit(mlb_batter_x_train, mlb_batter_y_train)

    mlb_batter_y_predict = baseball_batter_model.predict(mlb_batter_x_test)

    plt.scatter(mlb_batter_y_test, mlb_batter_y_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test))

    mlb_batter_weights = baseball_batter_model.coef_

    cpbl_batter_x = cpbl_csv[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    cpbl_batter_y = cpbl_csv[["hr"]]

    cpbl_batter_predict = baseball_batter_model.predict(cpbl_batter_x)
    plt.scatter(cpbl_batter_y, cpbl_batter_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted CPBL")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y))

    cpbl_data = cpbl_csv
    x_data = torch.FloatTensor(np.array(cpbl_batter_x))
    y_data = torch.FloatTensor(np.array(cpbl_batter_y))

    epochs = 30000

    targets_data = mlb_batter_weights.T
    targets_df = pd.DataFrame(data=targets_data)
    targets_df.columns = ["targets"]
    W = torch.Tensor(np.array(mlb_batter_weights).T)
    W = W.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-6)

    for epoch in range(epochs + 1):
        hypothesis = x_data.matmul(W) + b

        cost = torch.mean((hypothesis - y_data) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(
            "Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} ".format(
                epoch, epochs, hypothesis.squeeze().detach(), cost.item()
            )
        )
        if epoch == epochs:
            hypo = hypothesis.detach().numpy()

    hypo = np.ravel(hypo, order="C")
    total = 0
    for s in range(len(hypo)):
        total += hypo[s]

    cpbl_batter_y = np.array(cpbl_batter_y)
    cpbl_batter_y = np.ravel(cpbl_batter_y, order="C")
    total2 = 0
    for p in range(len(cpbl_batter_y)):
        total2 += cpbl_batter_y[p]

    cpbl_score = 1 - (abs(total - total2) / (total2))

    print(
        "MLB模型評估MLB選手時的準確度 =                {:.6f}".format(
            baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test)
        )
    )
    print(
        "MLB模型不調整評估CPBL選手時的準確度 =   {:.6f}".format(
            baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y)
        )
    )
    print(
        "MLB模型調整後評估CPBL選手時的準確度 =        {:.6f}".format(
            np.mean(cpbl_score)
        )
    )
    print("計算的最終權重 = ")
    print(W)

    cpbl_predictions = pd.DataFrame(
        {"Actual hr": cpbl_batter_y.flatten(), "Predicted hr": hypo}
    )

    print(cpbl_predictions)

    if playername in cpbl_csv["player"].values:
        b_ex = cpbl_csv[
            [
                "player",
                "team",
                "league",
                "year",
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        print(b_ex[b_ex["player"] == playername])
        print("\n\n")
        b_ex2 = b_ex[b_ex.player == playername]
        input_b = b_ex2[
            [
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        output_b = baseball_batter_model.predict(input_b)
        print(output_b)


# ------------------------------------------------------------------------------------------------------------------------------------------------


def obp(playername):
    batting_info = mlb_battingstats[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]

    mlb_batter_x = batting_info[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    mlb_batter_y = batting_info[["obp"]]
    mlb_batter_x_train, mlb_batter_x_test, mlb_batter_y_train, mlb_batter_y_test = (
        train_test_split(mlb_batter_x, mlb_batter_y, train_size=0.8, test_size=0.2)
    )

    baseball_batter_model = LinearRegression()
    baseball_batter_model.fit(mlb_batter_x_train, mlb_batter_y_train)

    mlb_batter_y_predict = baseball_batter_model.predict(mlb_batter_x_test)

    plt.scatter(mlb_batter_y_test, mlb_batter_y_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test))

    mlb_batter_weights = baseball_batter_model.coef_

    cpbl_batter_x = cpbl_csv[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    cpbl_batter_y = cpbl_csv[["obp"]]

    cpbl_batter_predict = baseball_batter_model.predict(cpbl_batter_x)
    plt.scatter(cpbl_batter_y, cpbl_batter_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted CPBL")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y))

    cpbl_data = cpbl_csv
    x_data = torch.FloatTensor(np.array(cpbl_batter_x))
    y_data = torch.FloatTensor(np.array(cpbl_batter_y))

    epochs = 30000

    targets_data = mlb_batter_weights.T
    targets_df = pd.DataFrame(data=targets_data)
    targets_df.columns = ["targets"]
    W = torch.Tensor(np.array(mlb_batter_weights).T)
    W = W.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-6)

    for epoch in range(epochs + 1):
        hypothesis = x_data.matmul(W) + b

        cost = torch.mean((hypothesis - y_data) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(
            "Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} ".format(
                epoch, epochs, hypothesis.squeeze().detach(), cost.item()
            )
        )
        if epoch == epochs:
            hypo = hypothesis.detach().numpy()

    hypo = np.ravel(hypo, order="C")
    total = 0
    for s in range(len(hypo)):
        total += hypo[s]

    cpbl_batter_y = np.array(cpbl_batter_y)
    cpbl_batter_y = np.ravel(cpbl_batter_y, order="C")
    total2 = 0
    for p in range(len(cpbl_batter_y)):
        total2 += cpbl_batter_y[p]

    cpbl_score = 1 - (abs(total - total2) / (total2))

    print(
        "MLB模型評估MLB選手時的準確度 =                {:.6f}".format(
            baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test)
        )
    )
    print(
        "MLB模型不調整評估CPBL選手時的準確度 =   {:.6f}".format(
            baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y)
        )
    )
    print(
        "MLB模型調整後評估CPBL選手時的準確度 =        {:.6f}".format(
            np.mean(cpbl_score)
        )
    )
    print("計算的最終權重 = ")
    print(W)

    cpbl_predictions = pd.DataFrame(
        {"Actual obp": cpbl_batter_y.flatten(), "Predicted obp": hypo}
    )

    print(cpbl_predictions)

    if playername in cpbl_csv["player"].values:
        b_ex = cpbl_csv[
            [
                "player",
                "team",
                "league",
                "year",
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        print(b_ex[b_ex["player"] == playername])
        print("\n\n")
        b_ex2 = b_ex[b_ex.player == playername]
        input_b = b_ex2[
            [
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        output_b = baseball_batter_model.predict(input_b)
        print(output_b)


# --------------------------------------------------------------------------------------------------------------------------------------------------


def pa(playername):
    batting_info = mlb_battingstats[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]

    mlb_batter_x = batting_info[
        [
            "g",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    mlb_batter_y = batting_info[["pa"]]
    mlb_batter_x_train, mlb_batter_x_test, mlb_batter_y_train, mlb_batter_y_test = (
        train_test_split(mlb_batter_x, mlb_batter_y, train_size=0.8, test_size=0.2)
    )

    baseball_batter_model = LinearRegression()
    baseball_batter_model.fit(mlb_batter_x_train, mlb_batter_y_train)

    mlb_batter_y_predict = baseball_batter_model.predict(mlb_batter_x_test)

    plt.scatter(mlb_batter_y_test, mlb_batter_y_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test))

    mlb_batter_weights = baseball_batter_model.coef_

    cpbl_batter_x = cpbl_csv[
        [
            "g",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    cpbl_batter_y = cpbl_csv[["pa"]]

    cpbl_batter_predict = baseball_batter_model.predict(cpbl_batter_x)
    plt.scatter(cpbl_batter_y, cpbl_batter_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted CPBL")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y))

    cpbl_data = cpbl_csv
    x_data = torch.FloatTensor(np.array(cpbl_batter_x))
    y_data = torch.FloatTensor(np.array(cpbl_batter_y))

    epochs = 30000

    targets_data = mlb_batter_weights.T
    targets_df = pd.DataFrame(data=targets_data)
    targets_df.columns = ["targets"]
    W = torch.Tensor(np.array(mlb_batter_weights).T)
    W = W.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-6)

    for epoch in range(epochs + 1):
        hypothesis = x_data.matmul(W) + b

        cost = torch.mean((hypothesis - y_data) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(
            "Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} ".format(
                epoch, epochs, hypothesis.squeeze().detach(), cost.item()
            )
        )
        if epoch == epochs:
            hypo = hypothesis.detach().numpy()

    hypo = np.ravel(hypo, order="C")
    total = 0
    for s in range(len(hypo)):
        total += hypo[s]

    cpbl_batter_y = np.array(cpbl_batter_y)
    cpbl_batter_y = np.ravel(cpbl_batter_y, order="C")
    total2 = 0
    for p in range(len(cpbl_batter_y)):
        total2 += cpbl_batter_y[p]

    cpbl_score = 1 - (abs(total - total2) / (total2))

    print(
        "MLB模型評估MLB選手時的準確度 =                {:.6f}".format(
            baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test)
        )
    )
    print(
        "MLB模型不調整評估CPBL選手時的準確度 =   {:.6f}".format(
            baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y)
        )
    )
    print(
        "MLB模型調整後評估CPBL選手時的準確度 =        {:.6f}".format(
            np.mean(cpbl_score)
        )
    )
    print("計算的最終權重 = ")
    print(W)

    cpbl_predictions = pd.DataFrame(
        {"Actual pa": cpbl_batter_y.flatten(), "Predicted pa": hypo}
    )

    print(cpbl_predictions)

    if playername in cpbl_csv["player"].values:
        b_ex = cpbl_csv[
            [
                "player",
                "team",
                "league",
                "year",
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        print(b_ex[b_ex["player"] == playername])
        print("\n\n")
        b_ex2 = b_ex[b_ex.player == playername]
        input_b = b_ex2[
            [
                "g",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        output_b = baseball_batter_model.predict(input_b)
        print(output_b)


# -----------------------------------------------------------------------------------------------------------------------------------------


def rbi(playername):
    batting_info = mlb_battingstats[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]

    mlb_batter_x = batting_info[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    mlb_batter_y = batting_info[["rbi"]]
    mlb_batter_x_train, mlb_batter_x_test, mlb_batter_y_train, mlb_batter_y_test = (
        train_test_split(mlb_batter_x, mlb_batter_y, train_size=0.8, test_size=0.2)
    )

    baseball_batter_model = LinearRegression()
    baseball_batter_model.fit(mlb_batter_x_train, mlb_batter_y_train)

    mlb_batter_y_predict = baseball_batter_model.predict(mlb_batter_x_test)

    plt.scatter(mlb_batter_y_test, mlb_batter_y_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test))

    mlb_batter_weights = baseball_batter_model.coef_

    cpbl_batter_x = cpbl_csv[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    cpbl_batter_y = cpbl_csv[["rbi"]]

    cpbl_batter_predict = baseball_batter_model.predict(cpbl_batter_x)
    plt.scatter(cpbl_batter_y, cpbl_batter_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted CPBL")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y))

    cpbl_data = cpbl_csv
    x_data = torch.FloatTensor(np.array(cpbl_batter_x))
    y_data = torch.FloatTensor(np.array(cpbl_batter_y))

    epochs = 30000

    targets_data = mlb_batter_weights.T
    targets_df = pd.DataFrame(data=targets_data)
    targets_df.columns = ["targets"]
    W = torch.Tensor(np.array(mlb_batter_weights).T)
    W = W.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-6)

    for epoch in range(epochs + 1):
        hypothesis = x_data.matmul(W) + b

        cost = torch.mean((hypothesis - y_data) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(
            "Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} ".format(
                epoch, epochs, hypothesis.squeeze().detach(), cost.item()
            )
        )
        if epoch == epochs:
            hypo = hypothesis.detach().numpy()

    hypo = np.ravel(hypo, order="C")
    total = 0
    for s in range(len(hypo)):
        total += hypo[s]

    cpbl_batter_y = np.array(cpbl_batter_y)
    cpbl_batter_y = np.ravel(cpbl_batter_y, order="C")
    total2 = 0
    for p in range(len(cpbl_batter_y)):
        total2 += cpbl_batter_y[p]

    cpbl_score = 1 - (abs(total - total2) / (total2))

    print(
        "MLB模型評估MLB選手時的準確度 =                {:.6f}".format(
            baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test)
        )
    )
    print(
        "MLB模型不調整評估CPBL選手時的準確度 =   {:.6f}".format(
            baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y)
        )
    )
    print(
        "MLB模型調整後評估CPBL選手時的準確度 =        {:.6f}".format(
            np.mean(cpbl_score)
        )
    )
    print("計算的最終權重 = ")
    print(W)

    cpbl_predictions = pd.DataFrame(
        {"Actual rbi": cpbl_batter_y.flatten(), "Predicted rbi": hypo}
    )

    print(cpbl_predictions)

    if playername in cpbl_csv["player"].values:
        b_ex = cpbl_csv[
            [
                "player",
                "team",
                "league",
                "year",
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        print(b_ex[b_ex["player"] == playername])
        print("\n\n")
        b_ex2 = b_ex[b_ex.player == playername]
        input_b = b_ex2[
            [
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        output_b = baseball_batter_model.predict(input_b)
        print(output_b)


# ---------------------------------------------------------------------------------------------------------------------------------------------------


def r(playername):
    batting_info = mlb_battingstats[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]

    mlb_batter_x = batting_info[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    mlb_batter_y = batting_info[["r"]]
    mlb_batter_x_train, mlb_batter_x_test, mlb_batter_y_train, mlb_batter_y_test = (
        train_test_split(mlb_batter_x, mlb_batter_y, train_size=0.8, test_size=0.2)
    )

    baseball_batter_model = LinearRegression()
    baseball_batter_model.fit(mlb_batter_x_train, mlb_batter_y_train)

    mlb_batter_y_predict = baseball_batter_model.predict(mlb_batter_x_test)

    plt.scatter(mlb_batter_y_test, mlb_batter_y_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test))

    mlb_batter_weights = baseball_batter_model.coef_

    cpbl_batter_x = cpbl_csv[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    cpbl_batter_y = cpbl_csv[["r"]]

    cpbl_batter_predict = baseball_batter_model.predict(cpbl_batter_x)
    plt.scatter(cpbl_batter_y, cpbl_batter_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted CPBL")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y))

    cpbl_data = cpbl_csv
    x_data = torch.FloatTensor(np.array(cpbl_batter_x))
    y_data = torch.FloatTensor(np.array(cpbl_batter_y))

    epochs = 30000

    targets_data = mlb_batter_weights.T
    targets_df = pd.DataFrame(data=targets_data)
    targets_df.columns = ["targets"]
    W = torch.Tensor(np.array(mlb_batter_weights).T)
    W = W.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-6)

    for epoch in range(epochs + 1):
        hypothesis = x_data.matmul(W) + b

        cost = torch.mean((hypothesis - y_data) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(
            "Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} ".format(
                epoch, epochs, hypothesis.squeeze().detach(), cost.item()
            )
        )
        if epoch == epochs:
            hypo = hypothesis.detach().numpy()

    hypo = np.ravel(hypo, order="C")
    total = 0
    for s in range(len(hypo)):
        total += hypo[s]

    cpbl_batter_y = np.array(cpbl_batter_y)
    cpbl_batter_y = np.ravel(cpbl_batter_y, order="C")
    total2 = 0
    for p in range(len(cpbl_batter_y)):
        total2 += cpbl_batter_y[p]

    cpbl_score = 1 - (abs(total - total2) / (total2))

    print(
        "MLB模型評估MLB選手時的準確度 =                {:.6f}".format(
            baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test)
        )
    )
    print(
        "MLB模型不調整評估CPBL選手時的準確度 =   {:.6f}".format(
            baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y)
        )
    )
    print(
        "MLB模型調整後評估CPBL選手時的準確度 =        {:.6f}".format(
            np.mean(cpbl_score)
        )
    )
    print("計算的最終權重 = ")
    print(W)

    cpbl_predictions = pd.DataFrame(
        {"Actual r": cpbl_batter_y.flatten(), "Predicted r": hypo}
    )

    print(cpbl_predictions)

    if playername in cpbl_csv["player"].values:
        b_ex = cpbl_csv[
            [
                "player",
                "team",
                "league",
                "year",
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        print(b_ex[b_ex["player"] == playername])
        print("\n\n")
        b_ex2 = b_ex[b_ex.player == playername]
        input_b = b_ex2[
            [
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        output_b = baseball_batter_model.predict(input_b)
        print(output_b)


# -----------------------------------------------------------------------------------------------------------------------------------------------


def slg(playername):
    batting_info = mlb_battingstats[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "slg",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]

    mlb_batter_x = batting_info[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    mlb_batter_y = batting_info[["slg"]]
    mlb_batter_x_train, mlb_batter_x_test, mlb_batter_y_train, mlb_batter_y_test = (
        train_test_split(mlb_batter_x, mlb_batter_y, train_size=0.8, test_size=0.2)
    )

    baseball_batter_model = LinearRegression()
    baseball_batter_model.fit(mlb_batter_x_train, mlb_batter_y_train)

    mlb_batter_y_predict = baseball_batter_model.predict(mlb_batter_x_test)

    plt.scatter(mlb_batter_y_test, mlb_batter_y_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test))

    mlb_batter_weights = baseball_batter_model.coef_

    cpbl_batter_x = cpbl_csv[
        [
            "g",
            "pa",
            "ab",
            "h",
            "2b",
            "3b",
            "hr",
            "rbi",
            "r",
            "bb",
            "ibb",
            "hbp",
            "so",
            "avg",
            "obp",
            "ops",
            "sb",
            "cs",
            "sh",
            "sf",
        ]
    ]
    cpbl_batter_y = cpbl_csv[["slg"]]

    cpbl_batter_predict = baseball_batter_model.predict(cpbl_batter_x)
    plt.scatter(cpbl_batter_y, cpbl_batter_predict, alpha=0.4)
    plt.xlabel("Real")
    plt.ylabel("Predicted CPBL")
    plt.title("MULTIPLE LINEAR REGRESSION")
    plt.show()

    print(baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y))

    cpbl_data = cpbl_csv
    x_data = torch.FloatTensor(np.array(cpbl_batter_x))
    y_data = torch.FloatTensor(np.array(cpbl_batter_y))

    epochs = 30000

    targets_data = mlb_batter_weights.T
    targets_df = pd.DataFrame(data=targets_data)
    targets_df.columns = ["targets"]
    W = torch.Tensor(np.array(mlb_batter_weights).T)
    W = W.requires_grad_(True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=1e-6)

    for epoch in range(epochs + 1):
        hypothesis = x_data.matmul(W) + b

        cost = torch.mean((hypothesis - y_data) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        print(
            "Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} ".format(
                epoch, epochs, hypothesis.squeeze().detach(), cost.item()
            )
        )
        if epoch == epochs:
            hypo = hypothesis.detach().numpy()

    hypo = np.ravel(hypo, order="C")
    total = 0
    for s in range(len(hypo)):
        total += hypo[s]

    cpbl_batter_y = np.array(cpbl_batter_y)
    cpbl_batter_y = np.ravel(cpbl_batter_y, order="C")
    total2 = 0
    for p in range(len(cpbl_batter_y)):
        total2 += cpbl_batter_y[p]

    cpbl_score = 1 - (abs(total - total2) / (total2))

    print(
        "MLB模型評估MLB選手時的準確度 =                {:.6f}".format(
            baseball_batter_model.score(mlb_batter_x_test, mlb_batter_y_test)
        )
    )
    print(
        "MLB模型不調整評估CPBL選手時的準確度 =   {:.6f}".format(
            baseball_batter_model.score(cpbl_batter_x, cpbl_batter_y)
        )
    )
    print(
        "MLB模型調整後評估CPBL選手時的準確度 =        {:.6f}".format(
            np.mean(cpbl_score)
        )
    )
    print("計算的最終權重 = ")
    print(W)

    cpbl_predictions = pd.DataFrame(
        {"Actual slg": cpbl_batter_y.flatten(), "Predicted slg": hypo}
    )

    print(cpbl_predictions)

    if playername in cpbl_csv["player"].values:
        b_ex = cpbl_csv[
            [
                "player",
                "team",
                "league",
                "year",
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "slg",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        print(b_ex[b_ex["player"] == playername])
        print("\n\n")
        b_ex2 = b_ex[b_ex.player == playername]
        input_b = b_ex2[
            [
                "g",
                "pa",
                "ab",
                "h",
                "2b",
                "3b",
                "hr",
                "rbi",
                "r",
                "bb",
                "ibb",
                "hbp",
                "so",
                "avg",
                "obp",
                "ops",
                "sb",
                "cs",
                "sh",
                "sf",
            ]
        ]
        output_b = baseball_batter_model.predict(input_b)
        print(output_b)
