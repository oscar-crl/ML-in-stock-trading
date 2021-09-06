import sklearn.metrics as skm
import pandas as pd


def compare_models():
    actual_prices = pd.read_csv('results/actu-prices.csv')
    base_pred_prices = pd.read_csv('results/base-model/price-base-model.csv')
    improved_pred_prices = pd.read_csv('results/improved-model/price-improved-model.csv')
    improved_epoch_pred_prices = pd.read_csv('results/improved-model/epoch/price-imp-epoch.csv')

    actual_prices = actual_prices.Price
    base_pred_prices = base_pred_prices.Price
    improved_pred_prices = improved_pred_prices.Price
    improved_epoch_pred_prices = improved_epoch_pred_prices.Price

    print("Explained variance score")
    print(f"Base model: {skm.explained_variance_score(actual_prices, base_pred_prices)}")
    print(f"Improved model: {skm.explained_variance_score(actual_prices, improved_pred_prices)}\n")

    print("Max error")
    print(f"Base model: {skm.max_error(actual_prices, base_pred_prices)}")
    print(f"Improved model: {skm.max_error(actual_prices, improved_pred_prices)}\n")

    print("Mean absolute percentage error")
    print(f"Base model: {skm.mean_absolute_percentage_error(actual_prices, base_pred_prices) * 100}")
    print(f"Improved model: {skm.mean_absolute_percentage_error(actual_prices, improved_pred_prices) * 100}\n")

    print("Mean square error")
    print(f"Base model: {skm.mean_squared_error(actual_prices, base_pred_prices)}")
    print(f"Improved model: {skm.mean_squared_error(actual_prices, improved_pred_prices)}\n")

    metrics(actual_prices, improved_pred_prices)
    metrics(actual_prices, improved_epoch_pred_prices)


def metrics(actual_prices, pred_prices):
    print("Explained variance score")
    print(f"{skm.explained_variance_score(actual_prices, pred_prices)}\n")

    print("Max error")
    print(f"{skm.max_error(actual_prices, pred_prices)}\n")

    print("Mean absolute percentage error")
    print(f"{skm.mean_absolute_percentage_error(actual_prices, pred_prices) * 100}\n")

    print("Mean square error")
    print(f"{skm.mean_squared_error(actual_prices, pred_prices)}\n")


compare_models()