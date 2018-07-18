import sys, os
import logging

from PyQt5 import QtWidgets, QtCore
from pyqtgraph.parametertree import ParameterTree, Parameter

import xml.etree.ElementTree as ElementTree

import numpy as np

SETTINGS_FILE = "twiddle.ini"

parameters = [
    {'name': 'Output file', 'type': 'str', 'value': ''},
    {'name': 'Select output file...', 'type': 'action'},

    {'name': 'Movement', 'type': 'group', 'children': [
        {'name': 'Amplitude', 'type': 'float', 'value': 20.0, 'step': 5.0, 'suffix': 'deg'},
        {'name': 'Frequency', 'type': 'float', 'value': 1.0, 'step': 0.1, 'suffix': 'Hz'},
        {'name': 'Cycles', 'type': 'float', 'value': 3.0, 'step': 0.5},
        {'name': 'Wait before and after', 'type': 'float', 'value': 1.0, 'step': 0.5, 'suffix': 'sec'},
    ]},

    {'name': 'DAQ', 'type': 'group', 'children': [
        {'name': 'Input', 'type': 'group', 'children': [
            {'name': 'Sampling frequency', 'type': 'float', 'value': 1000.0, 'step': 500.0, 'siPrefix': True,
             'suffix': 'Hz'},
            {'name': 'SG0', 'type': 'str', 'value': 'Dev1/ai0'},
            {'name': 'SG1', 'type': 'str', 'value': 'Dev1/ai1'},
            {'name': 'SG2', 'type': 'str', 'value': 'Dev1/ai2'},
            {'name': 'SG3', 'type': 'str', 'value': 'Dev1/ai3'},
            {'name': 'SG4', 'type': 'str', 'value': 'Dev1/ai4'},
            {'name': 'SG5', 'type': 'str', 'value': 'Dev1/ai5'},
            {'name': 'Get calibration...', 'type': 'action'},
            {'name': 'Calibration file', 'type': 'str'},  # , 'readonly': True}
            {'name': 'Digital sampling frequency', 'type': 'float', 'value': 100000.0, 'step': 1000.0,
             'siPrefix': True, 'suffix': 'Hz'},
            {'name': 'Digital input port', 'type': 'str', 'value': 'Dev1/port0/line16:17'},
            {'name': 'PWM return line', 'type': 'int', 'value': 16},
            {'name': 'V3V pulse line', 'type': 'int', 'value': 17}
        ]},
        {'name': 'Output', 'type': 'group', 'children': [
            {'name': 'Digital port', 'type': 'str', 'value': 'Dev1/port0'},
            {'name': 'Inhibit line', 'type': 'int', 'value': 0},
            {'name': 'Enable line', 'type': 'int', 'value': 2},
            {'name': 'Counter name', 'type': 'str', 'value': 'Dev1/ctr0'}
        ]},
    ]},
    {'name': 'Motor', 'type': 'group', 'children': [
        {'name': 'Maximum speed', 'type': 'float', 'value': 400.0, 'step': 50.0, 'suffix': 'RPM'},
        {'name': 'Pulse frequency', 'type': 'float', 'value': 1000.0, 'step': 100.0, 'siPrefix': True,
         'suffix': 'Hz'},
        {'name': 'Sign convention', 'type': 'list', 'values': ['Left is positive', 'Left is negative', 'None'],
         'value': 'Left is positive'}
    ]}
]

class TwiddleSetupDialog(QtWidgets.QDialog):
    def __init__(self, parameters):
        super(TwiddleSetupDialog, self).__init__()

        self.params = Parameter.create(name='Parameters', type='group',
                                       children = parameters)
        self.paramtree = ParameterTree()
        self.paramtree.setParameters(self.params, showTop=False)

        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                                    QtWidgets.QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.paramtree)
        layout.addWidget(self.buttonBox)

        self.readSettings()

        self.params.child('DAQ', 'Input', 'Get calibration...').sigActivated.connect(self.getCalibration)
        self.params.child('Select output file...').sigActivated.connect(self.getOutputFile)

        self.setLayout(layout)

    def accept(self):
        self.writeSettings()
        self.loadCalibration()

        super(TwiddleSetupDialog, self).accept()

    def getOutputFile(self):
        outputFile = self.params['Output file']
        if not outputFile:
            outputFile = ""
        outputFile, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Choose output file", directory=outputFile,
                                                            filter="*.h5")
        if outputFile:
            self.params['Output file'] = outputFile

    def getCalibration(self):
        calibrationFile = self.params['DAQ', 'Input', 'Calibration file']
        if not calibrationFile:
            calibrationFile = ""
        calibrationFile, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose calibration file", directory=calibrationFile,
                                                            filter="*.cal")
        if calibrationFile:
            self.params['DAQ', 'Input', 'Calibration file'] = calibrationFile

    def loadCalibration(self):
        calibrationFile = self.params['DAQ', 'Input', 'Calibration file']
        if not calibrationFile:
            return
        if not os.path.exists(calibrationFile):
            raise IOError("Calibration file %s not found", calibrationFile)

        try:
            tree = ElementTree.parse(calibrationFile)
            cal = tree.getroot().find('Calibration')
            if cal is None:
                raise IOError('Not a calibration XML file')

            mat = []
            for ax in cal.findall('UserAxis'):
                txt = ax.get('values')
                row = [float(v) for v in txt.split()]
                mat.append(row)

        except IOError:
            logging.warning('Bad calibration file')
            return

        self.calibration = np.array(mat).T

    def readSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        settings.beginGroup("SetupDialog")
        self.resize(settings.value("size", type=QtCore.QSize, defaultValue=QtCore.QSize(800, 600)))
        self.move(settings.value("position", type=QtCore.QPoint, defaultValue=QtCore.QPoint(200, 200)))
        settings.endGroup()

        settings.beginGroup("Parameters")

        self.readParameters(settings, self.params)

        settings.endGroup()

    def writeSettings(self):
        settings = QtCore.QSettings(SETTINGS_FILE, QtCore.QSettings.IniFormat)

        logging.debug('Writing settings!')

        settings.beginGroup("SetupDialog")
        settings.setValue("size", self.size())
        settings.setValue("position", self.pos())
        settings.endGroup()

        settings.beginGroup("Parameters")
        self.writeParameters(settings, self.params)
        settings.endGroup()

    def writeParameters(self, settings, params):
        for ch in params:
            if ch.hasChildren():
                settings.beginGroup(ch.name())
                settings.setValue("Expanded", ch.opts['expanded'])
                self.writeParameters(settings, ch)
                settings.endGroup()
            elif ch.type() in ['float', 'int', 'list', 'str']:
                settings.setValue(ch.name(), ch.value())

    def readParameters(self, settings, params):
        for ch in params:
            if ch.hasChildren():
                settings.beginGroup(ch.name())
                expanded = settings.value("Expanded", defaultValue=False)
                ch.setOpts(expanded=expanded)

                self.readParameters(settings, ch)
                settings.endGroup()
            else:
                if ch.type() == 'float':
                    if settings.contains(ch.name()):
                        v = settings.value(ch.name(), type=float)
                        ch.setValue(v)
                elif ch.type() == 'int':
                    if settings.contains(ch.name()):
                        v = settings.value(ch.name(), type=int)
                        ch.setValue(v)
                elif ch.type() in ['str', 'list']:
                    if settings.contains(ch.name()):
                        ch.setValue(settings.value(ch.name(), type=str))





