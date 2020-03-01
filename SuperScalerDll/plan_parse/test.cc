#include "plan_parse.h"

int main()
{
    std::string cfg_path = "../plan/execution_plan/10.0.0.21-0.cfg";

    PlanTable table;
    table.readConfig(cfg_path);

    std::cout << "================================" << std::endl;
    std::cout << "displayPlanTable:" << std::endl;
    table.displayPlanTable();

    std::cout << "================================" << std::endl;
    std::cout << "hasPlan True and False:" << std::endl;
    std::cout << table.hasPlan("For_gradients/SuperScaler_SubgraphFCs/SuperScaler_Backward_SubgraphBpFCs/dense3_add_grad/tuple/control_dependency_1") << std::endl
              << table.hasPlan("tensor_11") << std::endl;

    std::cout << "================================" << std::endl;
    std::cout << "getPlan and displayPlan:" << std::endl;
    Plan plan = table.getPlan("For_gradients/SuperScaler_SubgraphFCs/SuperScaler_Backward_SubgraphBpFCs/dense3_add_grad/tuple/control_dependency_1");
    plan.displayPlan();

    std::cout << "================================" << std::endl;
    std::cout << "getFirstAllreducePlan and displayPlan:" << std::endl;
    plan = table.getFirstAllreducePlan();
    plan.displayPlan();
}
// g++ -o test test.cc plan_parse.cc
// ./test