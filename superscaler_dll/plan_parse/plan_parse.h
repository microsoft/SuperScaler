#ifndef PLAN_PARSE_H
#define PLAN_PARSE_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>

enum OperationType
{
    sendOT,
    receiveOT,
    allreduceOT
};

enum CommunicationType
{
    DefaultCT,
    PcieCT,
    RdmaCT
};

class Plan
{
public:
    Plan(std::string tensor, std::string tensorOT, std::string tensorCT);
    void addEndPoints(std::string endPoint);
    void displayPlan();
    OperationType getOperationType();
    CommunicationType getCommunicationType();
    std::vector<std::string> getEndPoints();

private:
    std::string tensorName;
    OperationType tensorOperationType;
    CommunicationType tensorCommunicationType;
    std::vector<std::string> endPoints;
};

class PlanTable
{
public:
    void readConfig(std::string configFilePath);
    void displayPlanTable();
    bool hasPlan(std::string tensorName);
    Plan getPlan(std::string tensorName);
    Plan getFirstAllreducePlan();
    std::string getSelfName();

private:
    std::string selfName;
    std::unordered_map<std::string, Plan> planTable;
};

#endif // PLAN_PARSE_H
