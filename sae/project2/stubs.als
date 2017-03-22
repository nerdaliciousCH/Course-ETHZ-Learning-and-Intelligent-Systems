/*
 * Static model: Signatures
 *
 * The model should contain the following (and potentially other) signatures.
 * If necessary, you have to make some of the signatures abstract and
 * make them extend other signatures.
 */

sig Aircraft { }

sig Airline { }

sig Airport { }

sig Booking { }

sig Class { }

sig Flight { }

sig Passenger { }

sig RoundTrip { }

sig Seat { }

sig Time { }

/*
 * Static model: Predicates
 */

// True iff t1 is strictly before t2.
pred isBefore[t1, t2: Time] { ... }

/*
 * Static model: Functions
 */

// Returns the departure time of the given flight.
fun getDeparture[f: Flight]: Time { ... }

// Returns the arrival time of the given flight.
fun getArrival[f: Flight]: Time { ... }

// Returns the airport the given flight departs from.
fun getOrigin[f: Flight]: Airport { ... }

// Returns the destination airport of the given flight. 
fun getDestination[f: Flight]: Airport { ... }

// Returns the first flight of the given booking.
fun getFirstFlight[b: Booking]: Flight { ... }

// Returns the last flight of the given booking.
fun getLastFlight[b: Booking]: Flight { ... }

// Returns all seats of the given aircraft. 
fun getSeats[a: Aircraft]: set Seat { ... }

// Returns all flights for which is given aircraft is used.
fun getFlights[a: Aircraft]: set Flight { ... }

// Returns all bookings booked by the given passenger.
fun getBookings[p: Passenger]: set Booking { ... }

// Returns all flights contained in the given booking.
fun getFlightsInBooking[b: Booking]: set Flight { ... }

/*
 * Dynamic model: Functions
 */

// Returns the state which comes after the given state.
fun getNextState[s: State]: State { ... } 

// Returns the location of the given passenger at the given time. 
fun getPassengerLocation[t: Time, p: Passenger]: PassengerLocation { ... }

// Returns the location of the given aircraft at the given time.
fun getAircraftLocation[t: Time, ac: Aircraft]: AircraftLocation { ... }

// Returns the time whose state the given State represents.
fun getTime[s: State]: Time { ... }
