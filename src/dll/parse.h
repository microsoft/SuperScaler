#ifndef HOROVOD_PARSE_H
#define HOROVOD_PARSE_H

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <utility>

inline std::vector<std::string> split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

class excution_operation{
public:
    //excution_operation() = default;
    void set_operation_type(std::string type_name) {operation_type = type_name;}
    void set_context_value(std::string context, std::vector<int> &target)
    {
        std::vector<std::string> vec = split(context, ",");

        for (size_t i = 0; i < vec.size(); i++)
        {
            target.push_back( std::stoi(vec[i]));
        }
    }
    void set_context_value(std::string context, std::vector<size_t> &target)
    {
        std::vector<std::string> vec = split(context, ",");

        for (size_t i = 0; i < vec.size(); i++)
        {
            target.push_back(std::stoul(vec[i]));
        }
    }
    void set_average(std::string context)
    {
        if (context == "true")
            average = true;
        else
            average = false;        
    }
    void set_context(std::string context_name, std::string context)
    {
        if(context_name == "send_target")
            set_context_value(context,send_target);
        else if(context_name == "send_address")
            set_context_value(context,send_address);
        else if(context_name == "send_length")
            set_context_value(context,send_length);
        else if(context_name == "receive_target")
            set_context_value(context,receive_target);
        else if(context_name == "receive_address")
            set_context_value(context,receive_address);
        else if(context_name == "receive_length")
            set_context_value(context,receive_length);
        else if(context_name == "average")
            set_average(context);
        else 
            return;
    }

    void show_context()
    {
        std::cout << "    " << "operation_type = " << operation_type << "\n";
        for(size_t i = 0; i < send_target.size(); i++)
            std::cout << "        " << "send_target[" << i << "] = " << send_target[i] << "\n";
        for(size_t i = 0; i < send_address.size(); i++)
            std::cout << "        " << "send_address[" << i << "] = " << send_address[i] << "\n";
        for(size_t i = 0; i < send_length.size(); i++)
            std::cout << "        " << "send_length[" << i << "] = " << send_length[i] << "\n";
        for(size_t i = 0; i < receive_target.size(); i++)
            std::cout << "        " << "receive_target[" << i << "] = " << receive_target[i] << "\n";
        for(size_t i = 0; i < receive_address.size(); i++)
            std::cout << "        " << "receive_address[" << i << "] = " << receive_address[i] << "\n";
        for(size_t i = 0; i < receive_length.size(); i++)
            std::cout << "        " << "receive_length[" << i << "] = " << receive_length[i] << "\n";
    }
    
    std::string operation_type;
    std::vector<int> send_target;
    std::vector<size_t> send_address;
    std::vector<size_t> send_length;
    std::vector<int> receive_target;
    std::vector<size_t> receive_address;
    std::vector<size_t> receive_length;
    bool average;
};

class plan{

public:

    void set_tensor_name(std::string name) {tensor_name = name;}
    void set_tensor_size(size_t size) {tensor_size = size;}
    void set_ip(std::vector<std::string> ip) {host_ip = ip;}
    void set_port(std::vector<std::string> port) {host_port = port;}
    void add_operation(std::string name){
        excution_operation new_operation;
        new_operation.set_operation_type(name);
        operation.push_back(new_operation);
    }
    void set_last_operation_contest(std::string context_name, std::string context) {
        operation.back().set_context(context_name,context);
    }

    void show_context()
    {
        std::cout << "tensor_name = " <<tensor_name << "\n";
        std::cout << "tensor_size = " <<tensor_size << "\n";
        std::cout << "IP = ";
        for(auto s : host_ip )
        {
            std::cout << s <<" ";
        }
        std::cout << "\n";

        std::cout << "port = ";
        for(auto s : host_port )
        {
            std::cout << s <<" ";
        }
        std::cout << "\n";
        
        for(size_t i = 0; i < operation.size(); i++)
        {
            operation[i].show_context();
        }
    }

    std::string tensor_name;
    size_t tensor_size;
    std::vector<excution_operation> operation;
    std::vector<std::string> host_ip;
    std::vector<std::string> host_port;
};

class CfgTable{

public:

    void parse_excution_plan();
    void parse_excution_plan(std::string);
    std::unordered_map<std::string, plan> cfg_table;

    void show_context()
    {
        std::cout << "Cfg table context" << "\n";
        for(auto cfg : cfg_table)
        {
            cfg.second.show_context();
        }
    }
};



#endif // HOROVOD_PARSE_H
