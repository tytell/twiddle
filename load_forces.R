library('rhdf5')
library('dplyr')

load_twiddle_data <- function(filename, cutoffmult=8.0) {
  h5file <- H5Fopen(filename)
  
  td <- data.frame(xtorque0 = h5file$Calibrated$xTorque,
                   ytorque0 = h5file$Calibrated$yTorque,
                   ztorque0 = h5file$Calibrated$zTorque)
  
  sampfreq <- h5readAttributes(h5file, '/Calibrated')$SampleFrequency
  waitbefore <- h5readAttributes(h5file, '/NominalStimulus')$WaitPrePost
  pretrigger <- h5readAttributes(h5file, '/ParameterTree/DAQ')$`Pretrigger duration`
  
  td$t <- seq(0, length=length(td$xtorque)) / sampfreq - waitbefore - pretrigger
  
  pos <- h5file$NominalStimulus$Position
  vel <- h5file$NominalStimulus$Velocity
  tout <- h5file$NominalStimulus$t
  
  td$pos <- approx(tout, pos, td$t, yleft=0.0, yright=0.0)$y
  td$vel <- approx(tout, pos, td$t, yleft=0.0, yright=0.0)$y
  
  stimattrs <- h5readAttributes(h5file, '/NominalStimulus')
  
  td$Frequency <- stimattrs$Frequency
  
  if ("PositionAmplitude" %in% names(stimattrs)) {
    td$Amplitude <- stimattrs$PositionAmplitude
  }
  else if ("Amplitude" %in% names(stimattrs)) {
    td$Amplitude <- stimattrs$Amplitude
  }
  else {
    td$Amplitude <- na
  }
  
  # subtract the bias
  bias <- td %>% dplyr::filter(t < 0) %>%
    select(ends_with("torque0")) %>%
    summarize_all(mean)
  
  td$xtorque0 = td$xtorque0 - bias$xtorque0
  td$ytorque0 = td$ytorque0 - bias$ytorque0
  td$ztorque0 = td$ztorque0 - bias$ztorque0
  
  # filter the data with a low pass filter
  cutoff <- cutoffmult * stimattrs$Frequency
  
  filtcoef <- signal::butter(9, cutoff / (0.5 * sampfreq), type="low")
  
  td$xtorque <- signal::filtfilt(filtcoef, td$xtorque0)
  td$ytorque <- signal::filtfilt(filtcoef, td$ytorque0)
  td$ztorque <- signal::filtfilt(filtcoef, td$ztorque0)
  
  td$power <- -td$ztorque * (td$vel * pi/180.0)
  
  td
}