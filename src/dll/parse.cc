#include "parse.h"

using section = std::vector<std::pair<std::string, std::vector<std::string>>>;

std::string string_to_operation_type(std::string line)
{
    if(line.find("(sync)", 0) != std::string::npos)
        return "sync";
    else if (line.find("(send_receive)", 0) != std::string::npos)
        return "send_receive";
    else if (line.find("(send)", 0) != std::string::npos)
        return "send";
    else if (line.find("(receive)", 0) != std::string::npos)
        return "receive";
    else if (line.find("(write)", 0) != std::string::npos)
        return "write";
    else if (line.find("(read)", 0) != std::string::npos)
        return "read";
    else
        return "None";
}

std::string string_to_type(std::string line)
{
    if(line.find("[plan]", 0) != std::string::npos)
        return "plan";
    else
        return "None";
}

section read_cfg(std::string filename)
{
    section options;
    std::ifstream stream(filename);
    if(!stream.is_open()) {
        std::cerr<< "couldn't open input file: " << filename << std::endl;
        exit(0);
    }

    std::string line;
    while (getline(stream, line)) {
        line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\t'), line.end());
        line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
	    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

        if (line.empty()) continue;
        //LOG(INFO, 0) << line;
        switch(line[0]) {
            case '[':
                options.push_back(std::make_pair(string_to_type(line),std::vector<std::string>{}));
                break;
            case '\0':
            case '#' :
            case ';' :
                break;
            default:
                options.back().second.push_back(line);
                break;
        }
    }
    return options;
}

void parse_plan(std::vector<std::string> context, plan& plan_)
{
    //std::vector<std::vector<std::string>> context_pre;
    for (size_t i = 0 ; i < context.size(); i++)
    {
        //LOG(INFO, 0) << context[i];
        switch(context[i][0]){
            case '(':
                plan_.add_operation(string_to_operation_type(context[i]));
                break;
            case '\0':
            case '#':
            case ';':
                break;
            default:
                std::vector<std::string> vec = split(context[i], "=");
                if (vec[0] == "tensor_name")
                {
                    plan_.set_tensor_name(vec[1]);
                    break;
                }
                else if (vec[0] == "tensor_size")
                {
                    plan_.set_tensor_size(stoul(vec[1]));
                    break;                    
                }
                else if (vec[0] == "ip")
                {
                    plan_.set_ip(split(vec[1], ","));
                }
                else if (vec[0] == "port")
                {
                    plan_.set_port(split(vec[1], ","));
                }
                else 
                {
                    plan_.set_last_operation_contest(vec[0], vec[1]);
                    break;                    
                }                
        }
    }
}

void CfgTable::parse_excution_plan()
{
    auto env_value = std::getenv("HOROVOD_CONFIGURE");
    std::string cfg_path;
    if (env_value != nullptr )
        cfg_path = std::string(env_value);
    else
        cfg_path = std::string("configure.cfg");
    
    section options = read_cfg(cfg_path);

    for (size_t i = 0; i < options.size(); i++)
    {
        if(options[i].first == "plan")
        {
            plan plan_;
            parse_plan(options[i].second,plan_);
            //plan_.show_context();
            cfg_table[plan_.tensor_name] = plan_;
        }
    }
    show_context();
}

void CfgTable::parse_excution_plan(std::string cfg_path)
{
    section options = read_cfg(cfg_path);

    for (size_t i = 0; i < options.size(); i++)
    {
        if(options[i].first == "plan")
        {
            plan plan_;
            parse_plan(options[i].second,plan_);
            //plan_.show_context();
            cfg_table[plan_.tensor_name] = plan_;
        }
    }
    show_context();
}