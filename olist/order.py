import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        """
        orders = self.data['orders'].copy()

        if is_delivered:
            orders = orders[orders['order_status'] == 'delivered']

        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

        orders['wait_time'] = (orders['order_delivered_customer_date'] -
                                orders['order_purchase_timestamp']).dt.days

        orders['expected_wait_time'] = (orders['order_estimated_delivery_date'] -
                                         orders['order_purchase_timestamp']).dt.days

        orders['delay_vs_expected'] = (orders['order_delivered_customer_date'] -
                                        orders['order_estimated_delivery_date']).dt.days.apply(
                                            lambda x: max(x, 0))

        return orders[['order_id', 'wait_time', 'expected_wait_time',
                        'delay_vs_expected', 'order_status']]

    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'].copy()

        reviews['dim_is_five_star'] = (reviews['review_score'] == 5).astype(int)
        reviews['dim_is_one_star'] = (reviews['review_score'] == 1).astype(int)

        return reviews[['order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score']]\
            .groupby('order_id').mean().reset_index()

    def get_number_items(self):
        """
        Returns a DataFrame with:
        order_id, number_of_items
        """
        order_items = self.data['order_items'].copy()

        return order_items.groupby('order_id').agg(
            number_of_items=('order_item_id', 'count')
        ).reset_index()

    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        order_items = self.data['order_items'].copy()

        return order_items.groupby('order_id').agg(
            number_of_sellers=('seller_id', 'nunique')
        ).reset_index()

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        order_items = self.data['order_items'].copy()

        return order_items.groupby('order_id').agg(
            price=('price', 'sum'),
            freight_value=('freight_value', 'sum')
        ).reset_index()

    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with:
        order_id, distance_seller_customer
        """
        data = self.data
        order_items = data['order_items']
        sellers = data['sellers']
        customers = data['customers']
        orders = data['orders']
        geo = data['geolocation']

        geo_mean = geo.groupby('geolocation_zip_code_prefix').agg(
            lat=('geolocation_lat', 'mean'),
            lng=('geolocation_lng', 'mean')
        ).reset_index()

        sellers_geo = sellers.merge(geo_mean, left_on='seller_zip_code_prefix',
                                     right_on='geolocation_zip_code_prefix')
        customers_geo = customers.merge(geo_mean, left_on='customer_zip_code_prefix',
                                         right_on='geolocation_zip_code_prefix')

        order_seller = order_items[['order_id', 'seller_id']].merge(
            sellers_geo[['seller_id', 'lat', 'lng']], on='seller_id')

        order_customer = orders[['order_id', 'customer_id']].merge(
            customers_geo[['customer_id', 'lat', 'lng']], on='customer_id')

        merged = order_seller.merge(order_customer, on='order_id',
                                     suffixes=('_seller', '_customer'))

        merged['distance_seller_customer'] = merged.apply(
            lambda r: haversine_distance(r['lat_seller'], r['lng_seller'],
                                          r['lat_customer'], r['lng_customer']), axis=1)

        return merged.groupby('order_id').agg(
            distance_seller_customer=('distance_seller_customer', 'mean')
        ).reset_index()

    def get_training_data(self,
                          is_delivered=True,
                          with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (without NaN), with the following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_items', 'number_of_sellers', 'price', 'freight_value']
        """
        training_data = self.get_wait_time(is_delivered)\
            .merge(self.get_review_score(), on='order_id')\
            .merge(self.get_number_items(), on='order_id')\
            .merge(self.get_number_sellers(), on='order_id')\
            .merge(self.get_price_and_freight(), on='order_id')

        if with_distance_seller_customer:
            training_data = training_data.merge(
                self.get_distance_seller_customer(), on='order_id')

        return training_data.dropna()