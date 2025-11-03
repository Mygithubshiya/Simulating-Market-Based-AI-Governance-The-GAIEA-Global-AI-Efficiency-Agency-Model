import matplotlib.pyplot as plt
import numpy as np

# --- All our assumptions are here, we have considered a rough hypothetical value ---
# --- to understand the simulation. These values can (and should) be changed  ---
# --- to reflect the current status of real-world AI models and their usage. ---


BASE_ANNUAL_ENERGY_GROWTH = 0.25      # Our guess: 25% growth in AI energy demand per year, unchecked.
BASE_ANNUAL_RENEWABLE_GROWTH = 1.5  # The grid gets a *little* greener on its own, maybe 1.5% a year.
BASE_ANNUAL_EFFICIENCY_GAIN = 0.08  # Tech gets 8% faster/cheaper anyway (like Moore's Law).

# Policy impact (this is how much we think the GAIEA "EER" rating *helps*)
MARKET_RENEWABLE_MULTIPLIER = 1.5   # Market pressure makes companies demand green energy.
MARKET_EFFICIENCY_MULTIPLIER = 0.07 # Market pressure makes companies build more efficient models.


EFFICIENCY_DAMPENING_FACTOR = 0.8  # Efficiency is great, but 10% efficiency gain doesn't mean 10% less energy.
                                 # We'll say 80% of the gain directly offsets energy growth.
WATER_EFFICIENCY_MULTIPLIER = 0.5  # Let's assume water cooling tech improves at half the rate of compute efficiency.This is where we need to focus as this parameter makes a huge differnce.

"""Emissions & "Business As Usual" (BAU)"""
EMISSIONS_FACTOR_TONS_PER_TWH = 300 
BAU_ANNUAL_ENERGY_GROWTH = 0.25     # BAU assumes the same unchecked 25% growth.
BAU_ANNUAL_RENEWABLE_GROWTH = 1.0   # BAU assumes the grid *slowly* gets greener without any AI pressure.


# These are the numbers we use to check if the final result is "bad".
# This is just for the "RERUN SUGGESTION" logic, not the EER grade.
PROBLEM_THRESHOLDS = {
    'emissions_tons': 25000,
    'renewable_percent': 60,
    'efficiency_multiplier': 5.0,
    'water_liters': 5.0
}

"""--- EER RATING RUBRIC ---
This is the "grading rubric" for the EER Rating (A-F).
 We check the final numbers against these lists.
e.g., for renewables, 80% is 3 points, 90% is 4 points, 95% is 5 points.
Note the matrix below explains the points to be affecting the in the 
calculation of the EER rating"""

EER_RATING_RUBRIC = {
    'renewable_pts':     [60,   70,   80,   90,   95], # Higher is better
    'efficiency_pts':    [5.0,  6.0,  8.0,  10.0, 12.0],# Higher is better
    'emissions_pts':     [25000, 15000, 10000, 5000, 2500], # Lower is better
    'water_pts':         [5.0,  3.0,   2.0,   1.0,  0.5]  # Lower is better
}
# Total max points = 5 * 4 categories = 20


# --- Helper Functions ---
# These are just little tools to make the 'main' function cleaner.

def get_simple_input(prompt, default):
    """A simple helper to get a number from the user."""
    user_input = input(f"{prompt} [default: {default}]: ")
    # If the user just hits Enter, we use the default value.
    return float(user_input) if user_input.strip() else default

def get_bool_input(prompt, default_bool):
    """A simple helper to get a 'yes' or 'no' from the user."""
    default_str = 'y' if default_bool else 'n'
    user_input = input(f"{prompt} (y/n) [default: {default_str}]: ").lower().strip()
    if not user_input:
        return default_bool
    return user_input == 'y'


# --- EER Grader Function ---

def calculate_eer_rating(final_metrics):
    """
    This is the "grader." It takes the final numbers and turns them
    into a letter grade (A-F) based on our 20-point rubric.
    """
    total_score = 0
    rubric = EER_RATING_RUBRIC # Getting the rubric from our assumptions

    # This is a small "inner" function to calculate the score for one category.
    # It's cleaner than writing the same loop four times.
    def get_score(value, thresholds, lower_is_better=False):
        score = 0
        if lower_is_better:
            # For emissions/water, we sort high-to-low (e.g., 25000, 15000...)
            # We check how many "hard" thresholds it can pass.
            for threshold in sorted(thresholds, reverse=True):
                if value <= threshold:
                    score += 1
                else:
                    break 
        else:
            # For renewable/efficiency, we sort low-to-high (e.g., 60, 70...)
            for threshold in sorted(thresholds):
                if value >= threshold:
                    score += 1
                else:
                    break
        return score

    # Now we just call our inner function for each category and add up the points.
    total_score += get_score(final_metrics['renewable'], rubric['renewable_pts'])
    total_score += get_score(final_metrics['efficiency'], rubric['efficiency_pts'])
    total_score += get_score(final_metrics['emissions'], rubric['emissions_pts'], lower_is_better=True)
    total_score += get_score(final_metrics['water'], rubric['water_pts'], lower_is_better=True)

    # Here it is trying to convert the total score (out of 20) to a letter grade.
    if total_score >= 19:
        return "A++"
    elif total_score >= 17:
        return "A"
    elif total_score >= 14:
        return "B"
    elif total_score >= 11:
        return "C"
    elif total_score >= 8:
        return "D"
    else:
        return "F"


# --- Simulation Core ---

def run_simulation(user_choices):
    """
    This is the main "engine" of the simulation.
    It takes the user's starting numbers and policy choices,
    then loops through each year to predict the future.
    """
    
    num_years = user_choices['years']
    # Considering tthe simulation from 2025
    years_range = list(range(2025, 2025 + num_years))
    
    # We create lists to hold the data for each year.
    # We "seed" these lists with the user's initial 2025 values.
    energy_use = [user_choices['initial_energy']]
    water_use = [user_choices['initial_water']]
    renewable_percent = [user_choices['initial_renewable']]
    efficiency_multiplier = [1.0] # We start at 1.0x efficiency
    
    emissions = []
    bau_emissions = [] # "Business As Usual" (what happens with no policy)
    
    # Calculate the emissions for our starting year (2025)
    initial_emissions = energy_use[0] * (1 - renewable_percent[0]/100) * EMISSIONS_FACTOR_TONS_PER_TWH
    emissions.append(initial_emissions)
    bau_emissions.append(initial_emissions)
    
    # Check if the policy is active. If not, market pressure is zero.
    market_pressure = 0
    if user_choices['audit_policy_active']:
        market_pressure = user_choices['market_pressure']
    
    # --- This is the main simulation loop ---
    # We loop from 1 up to the total number of years.
    # (The 0th year, 2025, is already set)
    for i in range(1, num_years):
        
        # 1. How much do renewables grow?
        #    It's the base grid improvement + market pressure from GAIEA.
        market_push_renew = market_pressure * MARKET_RENEWABLE_MULTIPLIER
        new_renewable = renewable_percent[i-1] + BASE_ANNUAL_RENEWABLE_GROWTH + market_push_renew
        renewable_percent.append(min(new_renewable, 95)) # We cap it at 95%
        
        # 2. How much more efficient does AI get?
        #    It's base tech progress + companies competing for a better "EER" rating.
        market_push_efficiency = market_pressure * MARKET_EFFICIENCY_MULTIPLIER
        total_efficiency_gain = BASE_ANNUAL_EFFICIENCY_GAIN + market_push_efficiency
        
        # Efficiency is multiplicative, so we multiply by (1 + gain)
        new_efficiency = efficiency_multiplier[i-1] * (1 + total_efficiency_gain)
        efficiency_multiplier.append(new_efficiency)
        
        # 3. How much does total energy use grow?
        #    It's the base demand for AI, but *dampened* by our new efficiency.
        efficiency_dampening = total_efficiency_gain * EFFICIENCY_DAMPENING_FACTOR
        net_energy_growth = max(0.05, BASE_ANNUAL_ENERGY_GROWTH - efficiency_dampening) # Keep at least 5% growth
        
        new_energy = energy_use[i-1] * (1 + net_energy_growth)
        energy_use.append(new_energy)

        # 4. How much water is used?
        #    We assume water use is tied to energy, but also gets more efficient.
        water_efficiency_gain = total_efficiency_gain * WATER_EFFICIENCY_MULTIPLIER
        
        last_water_per_twh = 0
        if energy_use[i-1] > 0:
            last_water_per_twh = water_use[i-1] / energy_use[i-1]
            
        new_water_per_twh = last_water_per_twh * (1 - water_efficiency_gain)
        new_water = new_energy * new_water_per_twh
        water_use.append(max(0, new_water)) # Water can't go below zero
        
        # 5. Calculate this year's emissions
        #    The grid-wide emissions factor also slowly improves (by 1% a year)
        improving_emissions_factor = EMISSIONS_FACTOR_TONS_PER_TWH * (0.99 ** i) 
        new_emissions = new_energy * (1 - renewable_percent[i]/100) * improving_emissions_factor
        emissions.append(new_emissions)
        
        # 6. Calculate "Business As Usual" (BAU) for comparison
        #    This is the "what if we do nothing" scenario.
        bau_energy = energy_use[0] * ((1 + BAU_ANNUAL_ENERGY_GROWTH) ** i)
        bau_renewable = user_choices['initial_renewable'] + (i * BAU_ANNUAL_RENEWABLE_GROWTH)
        
        bau_emission = bau_energy * (1 - min(bau_renewable, 50)/100) * EMISSIONS_FACTOR_TONS_PER_TWH
        bau_emissions.append(bau_emission)

   
    return {
        'years_range': years_range,
        'energy_use': energy_use,
        'water_use': water_use,
        'emissions': emissions,
        'renewable_percent': renewable_percent,
        'efficiency_multiplier': efficiency_multiplier,
        'bau_emissions': bau_emissions,
        'user_choices': user_choices
    }

# --- Reporting & Plotting ---

def print_policy_report(results):
    """
    This function acts like a "consultant." It takes the raw 'results'
    data and writes a human-friendly summary and report.
    """
    
    print("\n" + "="*70)
    print(" Here's Our Analysis & Policy Report ")
    print("="*70)

    final_year_index = len(results['years_range']) - 1
    user_choices = results['user_choices']
    
    final_emissions = results['emissions'][final_year_index]
    final_renewable = results['renewable_percent'][final_year_index]
    final_efficiency = results['efficiency_multiplier'][final_year_index]
    final_water = results['water_use'][final_year_index]
    
 
    final_metrics = {
        'emissions': final_emissions,
        'renewable': final_renewable,
        'efficiency': final_efficiency,
        'water': final_water
    }
    
    eer_grade = calculate_eer_rating(final_metrics)
    print("\n==============================================")
    print(f" PROJECTED EER RATING: {eer_grade}")
    print("==============================================")
    
    
    print(f"\nANALYSIS FOR YEAR {results['years_range'][-1]}:")
    print("-" * 50)
    print(f"Policy Status: GAIEA is {'ACTIVE' if user_choices['audit_policy_active'] else 'INACTIVE'}")
    if user_choices['audit_policy_active']:
        print(f"Market Pressure Factor: {user_choices['market_pressure']} / 5")
    
    print(f"\nProjected Emissions: {final_emissions:,.0f} tons CO2")
    print(f"Projected Water Use: {final_water:,.2f} Billion Liters")
    print(f"Projected Renewable %: {final_renewable:.1f}%")
    print(f"Projected Efficiency Gain: {final_efficiency:.2f}x")
    
    # Now, we check for problems.
    # We use this 'problems' list to decide what advice to give to the companies based on their usage, which will help them predict their future consumptions.
    problems = detect_problems(final_metrics)
    
       
    if problems:
        # If there are problems, we will try to generate recommendations.
        recommendations = generate_targeted_recommendations(problems, user_choices, eer_grade)
        print(f"\nREQUIRED POLICY ACTIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
            
        show_ideal_parameters(user_choices, problems) 
        
    else:
        print(f"\nEXCELLENT: Trajectory is sustainable! (Grade: {eer_grade})")
        print("   Continue with current GAIEA policy and market engagement.")

def detect_problems(final_metrics):
    
    problems = [] 
    
    if final_metrics['emissions'] > PROBLEM_THRESHOLDS['emissions_tons']:
        problems.append(f"Emissions critically high (>{final_metrics['emissions']:,.0f} tons)")
    
    if final_metrics['renewable'] < PROBLEM_THRESHOLDS['renewable_percent']:
        problems.append(f"Renewable adoption too slow (<{final_metrics['renewable']:.0f}%)")
    
    if final_metrics['efficiency'] < PROBLEM_THRESHOLDS['efficiency_multiplier']:
        problems.append(f"Efficiency gains insufficient (<{final_metrics['efficiency']:.1f}x)")

    if final_metrics['water'] > PROBLEM_THRESHOLDS['water_liters']:
        problems.append(f"Water consumption is too high (>{final_metrics['water']:.1f} B Liters)")
    
    return problems

def generate_targeted_recommendations(problems, user_choices, eer_grade):
    """
    #This function generates a "Required Policy Actions" list based on the
     advice depends on whether the policy was active or not.
    """
    recommendations = []
    is_poor_grade = eer_grade in ['D', 'F'] # Here we only check if the grade is bad, rest is not considered if acceptable
    
    if not user_choices['audit_policy_active']:
        ### If the policy was OFF, the main advice is to turn it ON, as our objective is to use the policy and check the difference after the implementation ofthe policy
        prefix = f"[URGENT: '{eer_grade}' RATING] " if is_poor_grade else""
        recommendations.append(f"{prefix}ACTIVATE POLICY: Establish the GAIEA to create 'EER' ratings.")
        recommendations.append("Mandate resource reporting (energy, water, carbon) for all frontier model training.")
    
    else:
        # The policy was ON, but we still got a bad grade.
        # This means the policy isn't strong enough, and we need to start working on it, making changes in the the parameters effecting the grade.
        prefix = f"[URGENT: '{eer_grade}' RATING] " if is_poor_grade else ""
        recommendations.append(f"{prefix}STRENGTHEN POLICY: GAIEA is active, but the Market Pressure ({user_choices['market_pressure']}/5) is too low.")
        
        # Give specific advice based on the *type* of problem
        if any("emissions" in p or "renewable" in p for p in problems):
            recommendations.append("Boost Market Demand: Launch public awareness campaigns for the 'EER' rating.")
            recommendations.append("Create Government Incentives: Offer tax breaks or priority contracts for companies using 'A' rated AI models.")

        if any("efficiency" in p for p in problems):
            recommendations.append("Spur Innovation: Fund an 'X-Prize' via GAIEA for hyper-efficient AI architectures.")
        
        if any("water" in p for p in problems):
            recommendations.append("Expand GAIEA Scope: Add water-use efficiency (e.g., water-free cooling) as a key part of the 'A' rating.")
    
    return recommendations

def show_ideal_parameters(user_choices, problems):
    """
    This function gives the user "hints" on how to get a better
    grade on their next run. It looks at the *problems* it found
    to give smarter, more specific advice.
    """
    print(f"\n--- RERUN SUGGESTION ---")
    print("To see a better grade, try rerunning with these adjustments:")
    
    # --- Policy Advice ---
    if not user_choices['audit_policy_active']:
        print(" - Activate the GAIEA Audit Policy: Set to 'y'")
    
    # If policy is on but pressure isn't maxed, suggest maxing it.
    elif user_choices['market_pressure'] < 5:
        print(f" - Increase Market Pressure: {user_choices['market_pressure']} -> 5 (Max out market forces)")
    
    
    
    if any("Renewable adoption too slow" in p for p in problems):
        if user_choices['initial_renewable'] < 50:
            print(f" - Increase Initial Renewables: {user_choices['initial_renewable']}% -> 50%+")
            print("   (Invest in renewables *before* the growth starts)")

    if any("Emissions critically high" in p for p in problems):
        if user_choices['initial_energy'] > 150:
            print(f" - Reduce Initial Energy Use: {user_choices['initial_energy']} TWh -> 150 TWh")
            print("   (Your starting energy use is very high, making an 'A' grade difficult)")

    if any("Water consumption is too high" in p for p in problems):
        if user_choices['initial_water'] > 2.0:
            print(f" - Reduce Initial Water Use: {user_choices['initial_water']} B Liters -> 2.0 B Liters")
            print("   (High starting water use is hurting your grade)")
            
    print("--------------------------")


def create_simple_plots(results):

###    It takes the 'results' and plots the values using matplotlib.
    
    years = results['years_range']
    
    # Creating 4 plots in a 2x2 grid to show how GAIEA can impact on the total consumption of water consumption and carbon emission. 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('GAIEA Climate Impact Simulation', fontsize=16, fontweight='bold')
    
    # This plot shows us the energy Consumption
    ax1.plot(years, results['energy_use'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title('AI Energy Consumption Forecast')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Energy (TWh)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(years)
    ax1.set_xticklabels(years, rotation=45)
    
    # This plot shows the Carbon Emissions Forecast 
    ax2.plot(years, results['emissions'], 'g-', linewidth=2, label='With GAIEA Policy', marker='s', markersize=4)
    ax2.plot(years, results['bau_emissions'], 'r--', linewidth=2, label='No Policy (BAU)')
    ax2.set_title('Carbon Emissions Forecast')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('CO2 Emissions (tons)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(years)
    ax2.set_xticklabels(years, rotation=45)
    
    # This plot shows us the renewable & efficiency graph 
    # For better realtion between % and x we create a "twin" axis that shares the same x-axis.
    ax3_twin = ax3.twinx() 
    line1 = ax3.plot(years, results['renewable_percent'], 'orange', linewidth=2, label='Renewable %', marker='^', markersize=4)
    line2 = ax3_twin.plot(years, results['efficiency_multiplier'], 'purple', linewidth=2, label='Efficiency', marker='d', markersize=4)
    ax3.set_title('Renewable Energy & Efficiency Gains')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Renewable Energy (%)', color='orange')
    ax3_twin.set_ylabel('Efficiency Multiplier (1.0 = 2025)', color='purple')
    
    # Here we have combined the "legends" from both axes to show them in one box, to show the differnce after usage of the policy
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(years)
    ax3.set_xticklabels(years, rotation=45)
    
    # This plot shows Water Consumption
    ax4.plot(years, results['water_use'], 'c-', linewidth=2, marker='p', markersize=4)
    ax4.set_title('Water Consumption Forecast')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Water (Billion Liters)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(years)
    ax4.set_xticklabels(years, rotation=45)
    

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()


def main():
   
    print("==============================================")
    print(" Welcome to the GAIEA Policy Simulator!")
    print("==============================================")
    print("Let's see if an 'Audit & Rating' policy can help the companies to calculate and get a rough idea of their usage and their impact on the planet.")
    print()
    
    ### Getting the companies consumption starting from the year 2025
    user_choices = {
        'years': int(get_simple_input("Number of years to simulate", 10)),
        'initial_energy': get_simple_input("Initial 2025 AI energy use (TWh/year)", 100),
        'initial_water': get_simple_input("Initial 2025 AI water use (Billion Liters/year)", 1.5),
        'initial_renewable': get_simple_input("Initial 2025 renewable energy (%)", 30)
    }
    
    print("-" * 50)
    
    ### Getting the policy choices
    user_choices['audit_policy_active'] = get_bool_input("Activate GAIEA Audit & EER Rating Policy?", default_bool=False)
    
    user_choices['market_pressure'] = 0
    if user_choices['audit_policy_active']:
        print("\nPolicy ACTIVE! Public ratings will drive market competition.")
        user_choices['market_pressure'] = get_simple_input("Market Pressure Factor (1-5, 5=high)", 3)
    else:
        print("\nPolicy INACTIVE. Simulating baseline growth.")
        
    ###print("\nStarting simulation...")
    
    results = run_simulation(user_choices)
    
    print_policy_report(results)
    
    create_simple_plots(results)
    
    print("\nSimulation complete.")


if __name__ == "__main__":
    main()