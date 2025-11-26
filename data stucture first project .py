def create_weather_records():
    records = []  
    n = int(input("Enter number of weather records to create: "))
    
    for i in range(n):
        print("\nEnter details for weather record " + str(i+1) + ":")
        record = dict()
        record['date'] = input("Enter date (YYYY-MM-DD): ")
        record['temperature'] = float(input("Enter temperature (in Celsius): "))
        record['city'] = input("Enter city name: ")
        records.append(record)
    
    print("\nWeather records created successfully!")
    return records


def display_weather_records(records):
    if not records:
        print("No weather records found.")
        return
    print("\nWeather Record Details:")
    for rec in records:
        print("Date: " + rec['date'] + ", Temperature: " + str(rec['temperature']) + " °C, City: " + rec['city'])


def delete_weather_record(records):
    if not records:
        print("No records available to delete.")
        return
    city_name = input("Enter the city name of the record to delete: ")
    for rec in records:
        if rec['city'].lower() == city_name.lower():
            records.remove(rec)
            print("Weather record for " + city_name + " deleted successfully!")
            return
    print("City not found. No record deleted.")


def retrieve_weather_record(records, city_name):
    if not records:
        print("No records available.")
        return
    found = False
    for rec in records:
        if rec['city'].lower() == city_name.lower():
            print("Record found: Date: " + rec['date'] + ", Temperature: " + str(rec['temperature']) + " °C, City: " + rec['city'])
            found = True
    if not found:
        print("City not found. No record retrieved.")



def Populate_array(years, cities):
    
    matrix = [[None for _ in range(len(cities))] for _ in range(len(years))]

    for i, year in enumerate(years):
        for j, city in enumerate(cities):
            temp = input("Enter temperature for " + city + " in " + year + " (or press Enter to skip): ")
            if temp.strip() == "":
                matrix[i][j] = None  
            else:
                matrix[i][j] = float(temp)
    return matrix


def row_major_access(matrix, years, cities):
    print("\nRow-Major Access (Year-wise):")
    for i, year in enumerate(years):
        line = year + ": "
        for j, city in enumerate(cities):
            line += city + "=" + str(matrix[i][j]) + " "
        print(line)


def column_major_access(matrix, years, cities):
    print("\nColumn-Major Access (City-wise):")
    for j, city in enumerate(cities):
        line = city + ": "
        for i, year in enumerate(years):
            line += year + "=" + str(matrix[i][j]) + " "
        print(line)



def analyze_complexity():
    print("\n--- Complexity Analysis ---")
    print("Insert (adding a record): O(1) per record")
    print("Delete (search and remove): O(n)")
    print("Retrieve (search by city): O(n)")
    print("Traverse (row/column access): O(n × m) for n years, m cities")
    print("Space Complexity: O(n × m) for matrix storage")



def main():
    records = [] 
    matrix = None
    years = []
    cities = []

    while True:
        choice = input(
            "\nMenu:\n"
            "1. Create Weather Records (CRUD)\n"
            "2. Display Weather Records\n"
            "3. Delete Weather Record\n"
            "4. Retrieve Data\n"
            "5. Populate array\n"
            "6. Row-Major Access\n"
            "7. Column-Major Access\n"
            "8. Analyze Complexity\n"
            "9. Exit\n"
            "Enter your choice: "
        )

        if choice == '1':
            records = create_weather_records()
        elif choice == '2':
            display_weather_records(records)
        elif choice == '3':
            delete_weather_record(records)
        elif choice == '4':
            city_name = input("Enter the city name to retrieve: ")
            retrieve_weather_record(records, city_name)
        elif choice == '5':
            years = input("Enter years separated by commas: ").split(",")
            cities = input("Enter cities separated by commas: ").split(",")
            matrix = Populate_array(years, cities)
        elif choice == '6':
            if matrix:
                row_major_access(matrix, years, cities)
            else:
                print("Matrix not created yet.")
        elif choice == '7':
            if matrix:
                column_major_access(matrix, years, cities)
            else:
                print("Matrix not created yet.")
        elif choice == '8':
            analyze_complexity()
        elif choice == '9':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")


if _name_ == "_main_":
    main()
