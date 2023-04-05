from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import MarkovChain as MC


def zad_3():
    # Create the model with edges specified as tuples (parent, child)
    dentist_model = BayesianNetwork([('Cavity', 'Toothache'),
                                     ('Cavity', 'Catch')])
    # Create tabular CPDs, values has to be 2-D array
    cpd_cav = TabularCPD('Cavity', 2, [[0.2], [0.8]],
                         state_names={'Cavity': [True, False]})
    cpd_too = TabularCPD('Toothache', 2, [[0.6, 0.1],
                                          [0.4, 0.9]],
                         evidence=['Cavity'], evidence_card=[2],
                         state_names={'Toothache': [True, False], 'Cavity': [True, False]})
    cpd_cat = TabularCPD('Catch', 2, [[0.9, 0.2],
                                      [0.1, 0.8]],
                         evidence=['Cavity'], evidence_card=[2],
                         state_names={'Catch': [True, False], 'Cavity': [True, False]})
    # Add CPDs to model
    dentist_model.add_cpds(cpd_cav, cpd_too, cpd_cat)

    print('Check model :', dentist_model.check_model())

    print('Independencies:\n', dentist_model.get_independencies())

    # Initialize inference algorithm
    dentist_infer = VariableElimination(dentist_model)

    # Some exemple queries
    q = dentist_infer.query(['Toothache'])
    print('P(Toothache) =\n', q)

    q = dentist_infer.query(['Cavity'])
    print('P(Cavity) =\n', q)

    q = dentist_infer.query(['Toothache'], evidence={'Cavity': True})
    print('P(Toothache | cavity) =\n', q)

    q = dentist_infer.query(['Toothache'], evidence={'Cavity': False})
    print('P(Toothache | ~cavity) =\n', q)

    # TODO 3.1

    q = dentist_infer.query(['Cavity'], evidence={'Toothache': True, 'Catch': False})
    print('P(Cavity | toothache, ~catch) =\n', q)

    # TODO 3.2

    # Version 1
    q = dentist_infer.query(['Cavity'], evidence={'Toothache': True, 'Catch': True})
    print('P(Cavity | toothache, catch) =\n', q)
    q = dentist_infer.query(['Cavity'], evidence={'Toothache': True, 'Catch': False})
    print('P(Cavity | toothache, ~catch) =\n', q)
    q = dentist_infer.query(['Cavity'], evidence={'Toothache': False, 'Catch': True})
    print('P(Cavity | ~toothache, catch) =\n', q)
    q = dentist_infer.query(['Cavity'], evidence={'Toothache': False, 'Catch': False})
    print('P(Cavity | ~toothache, ~catch) =\n', q)

    # Version 2
    a = dentist_infer.query(['Cavity', 'Toothache', 'Catch'])
    b = dentist_infer.query(['Toothache', 'Catch'])
    print('P(Cavity | Toothache, Catch) =\n', a/b)

    # TODO 3.3 state_names added in TabularCPD()


def zad_4():

    # TODO 4.1

    car_model = BayesianNetwork([('Battery', 'Radio'),
                                 ('Battery', 'Ignition'),
                                 ('Ignition', 'Starts'),
                                 ('Gas', 'Starts'),
                                 ('Starts', 'Moves')])

    cpd_bat = TabularCPD('Battery', 2, [[0.7], [0.3]],
                         state_names={'Battery': [True, False]})
    cpd_rad = TabularCPD('Radio', 2, [[0.9, 0.0],
                                      [0.1, 1.0]],
                         evidence=['Battery'], evidence_card=[2],
                         state_names={'Radio': [True, False], 'Battery': [True, False]})
    cpd_ign = TabularCPD('Ignition', 2, [[0.97, 0.0],
                                         [0.03, 1.0]],
                         evidence=['Battery'], evidence_card=[2],
                         state_names={'Ignition': [True, False], 'Battery': [True, False]})
    cpd_gas = TabularCPD('Gas', 2, [[0.5], [0.5]],
                         state_names={'Gas': [True, False]})
    cpd_sta = TabularCPD('Starts', 2, [[0.95, 0.0, 0.0, 0.0],
                                       [0.05, 1.0, 1.0, 1.0]],
                         evidence=['Ignition', 'Gas'], evidence_card=[2, 2],
                         state_names={'Starts': [True, False], 'Ignition': [True, False], 'Gas': [True, False]})
    cpd_mov = TabularCPD('Moves', 2, [[0.8, 0.0],
                                      [0.2, 1.0]],
                         evidence=['Starts'], evidence_card=[2],
                         state_names={'Moves': [True, False], 'Starts': [True, False]})

    car_model.add_cpds(cpd_bat, cpd_rad, cpd_ign, cpd_gas, cpd_sta, cpd_mov)

    print('Check model :', car_model.check_model())

    # print('Independencies:\n', car_model.get_independencies())

    car_infer = VariableElimination(car_model)

    # TODO 4.2

    q = car_infer.query(['Starts'], evidence={'Radio': True, 'Gas': True})
    print('P(Starts | radio, ~gas) =\n', q)

    # TODO 4.3

    q = car_infer.query(['Battery'], evidence={'Moves': True})
    print('P(Battery | moves) =\n', q)

    # TODO 4.4 and 4.5

    car_model.add_edge('NotIcyWeather', 'Starts')
    car_model.add_edge('StarterMotor', 'Starts')
    car_model.add_edge('Battery', 'StarterMotor')

    cpd_icy = TabularCPD('NotIcyWeather', 2, [[0.9], [0.1]],
                         state_names={'NotIcyWeather': [True, False]})
    cpd_mot = TabularCPD('StarterMotor', 2, [[0.95, 0.0],
                                             [0.05, 1.0]],
                         evidence=['Battery'], evidence_card=[2],
                         state_names={'StarterMotor': [True, False], 'Battery': [True, False]})
    cpd_sta = TabularCPD('Starts', 2, [[0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.15, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                         evidence=['Ignition', 'Gas', 'StarterMotor', 'NotIcyWeather'], evidence_card=[2, 2, 2, 2],
                         state_names={'Starts': [True, False], 'Ignition': [True, False], 'Gas': [True, False],
                                      'StarterMotor': [True, False], 'NotIcyWeather': [True, False]})

    car_model.add_cpds(cpd_icy, cpd_mot, cpd_sta)
    print('Check model :', car_model.check_model())

    # TODO 4.6
    q = car_infer.query(['Radio'], evidence={'Starts': False})
    print('P(Radio | ~starts) =\n', q)

    # TODO 4,7
    # Potrzeba 32 wartości do zapisania całej tabeli CPD dla Starts.

def zad_5():
    # weather_model = MC()
    # weather_model.add_variables_from(['Rain', 'Umbrella'], [2, 2])
    #
    # rain_tm = {0: {0: 0.7, 1: 0.3}, 1: {0: 0.3, 1: 0.7}}
    # weather_model.add_transition_model('Rain', rain_tm)
    # umbrella_tm = {0: {0: 0.8, 1: 0.1}, 1: {0: 0.2, 1: 0.9}}
    # weather_model.add_transition_model('Umbrella', umbrella_tm)
    #
    # from pgmpy.factors.discrete import State
    # weather_model.set_start_state([State('Rain', 0), State('Umbrella', 2)])
    #
    # weather_model.sample(size=5)

if __name__ == '__main__':
    zad_5()

