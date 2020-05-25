#!/usr/bin/env python3
import csv
import datetime
import pymysql.cursors

def open_connection(db, host="localhost", user="root"):
    return pymysql.connect(host=host, user=user, db=db, cursorclass=pymysql.cursors.DictCursor)

def close_connection(connection):
    connection.close()

def execute_mysql_command(connection, sql_command, arg=None):
    try:
        with connection.cursor() as cursor:
            if(arg == None):
                cursor.execute(sql_command)
            if(arg != None):
                cursor.execute(sql_command, arg)
            return cursor.fetchall()
    except:
        print("Error, mysql command: " + sql_command)
        return None

def get_dict_list(connection, sql_command):
    return execute_mysql_command(connection, sql_command)

def get_dict(connection, sql_command, keys, values):
    results = get_dict_list(connection, sql_command)
    dic = {}
    for line in results:
        key = []
        for _x in keys:
            key.append(line[_x])
        val = []
        for _y in values:
            val.append(line[_y])
        dic[tuple(key)] = tuple(val)
    return dic

def save_csv(connection, sql_command, csv_file):
    prt = ""
    results = execute_mysql_command(connection, sql_command)
    keys = []
    for key in results[0].keys():
        prt += str(key) + ","
        keys.append(key)
    prt = prt[:-1] + "\n"
    for data in results:
        for key in keys:
            prt += str(data[key]) + ","
        prt = prt[:-1] + "\n"
    f = open(csv_file, "w")
    f.write(prt)
    f.close()

def set_data_line_by_line(connection, tabname, data, date=True):
    key_tmp = "("
    val_tmp = "("
    for key in data.keys():
        if( len(key.strip().split()) > 1 ):
            key_tmp += "`" + key.strip() + "`,"
        if( len(key.strip().split()) == 1 ):
            key_tmp += key.strip() + ","
        val_tmp += "'" + str(data[key]).strip() +"',"
    if(date):
        now = datetime.datetime.utcnow()
        key_tmp += "`date (UTC)`)"
        val_tmp += "'{:s}')".format(now.strftime('%Y-%m-%d %H:%M:%S'))
    if(not date):
        key_tmp = key_tmp[:-1] + ")"
        val_tmp = val_tmp[:-1] + ")"
    sql = "insert ignore into " + tabname + " " + key_tmp + " values " + val_tmp
    results = execute_mysql_command(connection, sql)

def replace_data_line_by_line(connection, tabname, data, date=True):
    key_tmp = "("
    val_tmp = "("
    for key in data.keys():
        if( len(key.strip().split()) > 1 ):
            key_tmp += "`" + key.strip() + "`,"
        if( len(key.strip().split()) == 1 ):
            key_tmp += key.strip() + ","
        if( data[key] == "NULL"):
            val_tmp += "NULL,"
        else:
            val_tmp += "'" + str(data[key]).strip() +"',"
    if(date):
        now = datetime.datetime.utcnow()
        key_tmp += "`date (UTC)`)"
        val_tmp += "'{:s}')".format(now.strftime('%Y-%m-%d %H:%M:%S'))
    if(not date):
        key_tmp = key_tmp[:-1] + ")"
        val_tmp = val_tmp[:-1] + ")"
    sql = "replace into " + tabname + " " + key_tmp + " values " + val_tmp
    results = execute_mysql_command(connection, sql)

def set_data_from_csv(connection, tabname, csv_file, date=True):
    f = open(csv_file, "r")
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data = {}
        for i in range(len(row)):
            data[header[i].strip()] = row[i].strip()
        set_data_line_by_line(connection, tabname, data, date)
    f.close()
    connection.commit()

def replace_data_from_csv(connection, tabname, csv_file, date=True):
    f = open(csv_file, "r")
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data = {}
        for i in range(len(row)):
            data[header[i].strip()] = row[i].strip()
        replace_data_line_by_line(connection, tabname, data, date)
    f.close()
    connection.commit()

def show_example():
    print('---- open connection ----')
    print('connection = mysql.open_connection("HFMBPT")')
    print('tabname = "ground_state"')
    print('---- insert the data from csv ----')
    print('f = "energy.csv"')
    print('mysql.set_data_from_csv(connection, tabname, f)')
    print('')
    print('---- repalce the data from csv ----')
    print('f = "energy.csv"')
    print('mysql.replace_data_from_csv(connection, tabname, f)')
    print('')
    print('---- output to csv file ----')
    print('mysql.save_csv(connection, "select emax,e3max,`hw target (MeV)`,`Ehf (Mev)`,`EMBPT (MeV)` from ground_state where A=208", "text.csv")')
    print('')
    print('---- output to dictionary type ----')
    print('results=mysql.get_dict(connection, "select emax,e3max,`hw target (MeV)`,`Ehf (Mev)` from ground_state where A=208", ("emax", "e3max", "hw target (MeV)"), ("Ehf (MeV)"))')
    print('')
    print('---- close connection ----')
    print('mysql.close_connection(connection)')


if(__name__=="__main__"):
    connection = pymysql.connect(host="localhost",
            user="root", db="HFMBPT", cursorclass=pymysql.cursors.DictCursor)
    #tabname = "ground_state"
    #f = "energy.csv"
    #set_data_from_csv(connection, tabname, f)
    #save_csv(connection, "select emax,e3max,`hw target (MeV)`,`Ehf (Mev)`,`EMBPT (MeV)` from ground_state where A=208", "text.csv")
    connection.close()

