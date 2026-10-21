#!/usr/bin/env python3
"""
Nginx logs stored in MongoDB: stats including the top 10 IPs
"""


from pymongo import MongoClient


if __name__ == '__main__':

    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx

    # Get the total number of documents
    total_logs = collection.count_documents({})

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    method_counts = {method: collection.count_documents(
        {"method": method}) for method in methods}

    # Get the count of status check
    status_check_count = collection.count_documents(
        {"method": "GET", "path": "/status"})

    # Get the top 10 most present IPs
    top_ips = collection.aggregate([
        {"$group": {"_id": "$ip", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ])

    # Print the stats
    print(f"{total_logs} logs")
    print("Methods:")
    for method, count in method_counts.items():
        print(f"\tmethod {method}: {count}")
    print(f"{status_check_count} status check")
    print("IPs:")
    for ip in top_ips:
        print(f"\t{ip['_id']}: {ip['count']}")