from memory_profiler import profile

@profile
def process_data():
    data = []
    for i in range(10000):
        data.append(i ** 2)
    sum_data = sum(data)
    max_data = max(data)
    return sum_data, max_data

def main():
    result = process_data()
    print(result)

if __name__ == "__main__":
    main()
