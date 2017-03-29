/*
 * Static model: Signatures
 *
 * The model should contain the following (and potentially other) signatures.
 * If necessary, you have to make some of the signatures abstract and
 * make them extend other signatures.
 */

sig Aircraft {
	flights: set Flight
}{
	(all f1, f2: flights | isBefore[getDeparture[f1], getDeparture[f2]] => isBefore[getArrival[f1], getDeparture[f2]]) &&
	(no disj f1, f2: flights | getDeparture[f1] = getDeparture[f2] or getDeparture[f1] = getArrival[f2])
}

sig Airline { }

sig Airport { }

sig Booking { }

abstract sig Class { }

sig Flight {
	aircraft: one Aircraft,
	departure_time: one Time,
	arrival_time: one Time,
	origin: one Airport,
	destination: one Airport
}{
	isBefore[departure_time, arrival_time] &&
	origin != destination
}

fact {
	(all f: Flight, a: Aircraft | f in getFlights[a] <=> a = f.aircraft)
}

sig Passenger { }

sig RoundTrip extends Booking { }

sig Seat { }

sig Time { after: lone Time }{
	isBefore[this, after]  && !isBefore[after, this]
}
fact { one t: Time | Time = t.*after }

pred show {one a: Aircraft | #getFlights[a] > 1}
run show for 4
/*
 * Static model: Predicates
 */

// True iff t1 is strictly before t2.
pred isBefore[t1, t2: Time] { t2 in t1.^after }

/*
 * Static model: Functions
 */

// Returns the departure time of the given flight.
fun getDeparture[f: Flight]: Time { f.departure_time }
// Returns the arrival time of the given flight.
fun getArrival[f: Flight]: Time { f.arrival_time }
// Returns the airport the given flight departs from.
fun getOrigin[f: Flight]: Airport { f.origin }
// Returns the destination airport of the given flight. 
fun getDestination[f: Flight]: Airport { f.destination }
// Returns the first flight of the given booking.
//fun getFirstFlight[b: Booking]: Flight { ... }
// Returns the last flight of the given booking.
//fun getLastFlight[b: Booking]: Flight { ... }
// Returns all seats of the given aircraft. 
//fun getSeats[a: Aircraft]: set Seat { ... }
// Returns all flights for which is given aircraft is used.
fun getFlights[a: Aircraft]: set Flight { a.flights }
// Returns all bookings booked by the given passenger.
//fun getBookings[p: Passenger]: set Booking { ... }
// Returns all flights contained in the given booking.
//fun getFlightsInBooking[b: Booking]: set Flight { ... }
