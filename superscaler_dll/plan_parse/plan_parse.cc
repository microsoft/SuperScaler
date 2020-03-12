#include "plan_parse.h"

Plan::Plan(std::string tensor, std::string tensorOT, std::string tensorCT)
{
    tensorName = tensor;

    if (tensorOT == "_SCRecv")
    {
        tensorOperationType = receiveOT;
    }
    else if (tensorOT == "_SCSend")
    {
        tensorOperationType = sendOT;
    }
    else if (tensorOT == "_SCAllReduce")
    {
        tensorOperationType = allreduceOT;
    }

    if (tensorCT == "PCIE")
    {
        tensorCommunicationType = PcieCT;
    }
    else if (tensorCT == "RDMA")
    {
        tensorCommunicationType = RdmaCT;
    }
    else
    {
        tensorCommunicationType = DefaultCT;
    }
}

void Plan::addEndPoints(std::string endPoint)
{
    endPoints.push_back(endPoint);
}

void Plan::displayPlan()
{
    std::cout << tensorName << std::endl;
    std::cout << tensorOperationType << std::endl;
    std::cout << tensorCommunicationType << std::endl;
    for (int i = 0; i < endPoints.size(); i++)
    {
        std::cout << endPoints[i] << std::endl;
    }
}

OperationType Plan::getOperationType()
{
    return tensorOperationType;
}

CommunicationType Plan::getCommunicationType()
{
    return tensorCommunicationType;
}

std::vector<std::string> Plan::getEndPoints()
{
    return endPoints;
}

void PlanTable::readConfig(std::string configFilePath)
{
    std::ifstream configFile(configFilePath);
    if (!configFile.is_open())
    {
        std::cout << "Error: couldn't open config file:" << configFilePath << std::endl;
    }

    std::string serverName, gpuName, serverPort;
    configFile >> serverName >> gpuName >> serverPort;
    selfName = serverName + " " + gpuName + " " + serverPort;

    int tensorNum = 0;
    configFile >> tensorNum;

    for (int i = 0; i < tensorNum; i++)
    {
        std::string tensorName, tensorOT, tensorCT;
        configFile >> tensorName >> tensorOT >> tensorCT;

        Plan plan(tensorName, tensorOT, tensorCT);

        int endPointNum = 0;
        configFile >> endPointNum;
        for (int j = 0; j < endPointNum; j++)
        {
            std::string endPointIP, endPointGPU, endPointPort;
            configFile >> endPointIP >> endPointGPU >> endPointPort;
            plan.addEndPoints(endPointIP + " " + endPointGPU + " " + endPointPort);
        }

        planTable.insert({tensorName, plan});
    }

    configFile.close();
}

void PlanTable::displayPlanTable()
{
    std::cout << selfName << std::endl;
    for (auto &item : planTable)
    {
        auto plan = item.second;
        plan.displayPlan();
    }
}

bool PlanTable::hasPlan(std::string tensorName)
{
    return planTable.find(tensorName) != planTable.end();
}

Plan PlanTable::getPlan(std::string tensorName)
{
    auto plan = planTable.find(tensorName);
    return plan->second;
}

Plan PlanTable::getFirstAllreducePlan()
{
    for (auto &item : planTable)
    {
        auto plan = item.second;
        if (plan.getOperationType() == allreduceOT)
        {
            return plan;
        }
    }
}

std::string PlanTable::getSelfName()
{
    return selfName;
}