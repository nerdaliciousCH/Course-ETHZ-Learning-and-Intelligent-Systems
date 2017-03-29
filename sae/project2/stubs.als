/*
 * Static model: Signatures
 *
 * The model should contain the following (and potentially other) signatures.
 * If necessary, you have to make some of the signatures abstract and
 * make them extend other signatures.
 */

//open util/boolean

// possible that airport code is equal to booking ID
sig string {}

sig Aircraft {
	seats: some Seat,
	flights: set Flight
}{
	all disj f1, f2: flights | isBefore[getDeparture[f1], getDeparture[f2]] => isBefore[getArrival[f1], getDeparture[f2]]
	no disj f1, f2: flights | getDeparture[f1] = getDeparture[f2]
	//(all f1, f2: flights | ) // TODO check whether previous flight lands where next flight takes off
}

sig Airline {
	name: one string,
	aircrafts: set Aircraft,
	flight_routes: set Flight
}

sig Airport {
	code: one string
}
fact unique_airport_code { all disj A1, A2: Airport | A1.code != A2.code }

sig Booking {
	ID: one string,
	passengers: some Passenger,
	category: one Class,
	flights: some Flight
}{
	all disj f1, f2: flights | isBefore[getDeparture[f1], getDeparture[f2]] => isBefore[getArrival[f1], getDeparture[f2]]
	all disj f1, f2: flights | getDeparture[f1] != getDeparture[f2]
}

fact unique_booking_id{ all disj B1, B2: Booking | B1.ID != B2.ID }

sig RoundTrip extends Booking { }{ getOrigin[getFirstFlight[this]] = getDestination[getLastFlight[this]] }

abstract sig Class { }
one sig First_Class extends Class {}
one sig Business_Class extends Class {}
one sig Economy_Class extends Class {}

sig Flight {
	number: one string,
	operators: some Airline,
	aircraft: one Aircraft,
	bookings: set Booking,
	departure_time: one Time,
	arrival_time: one Time,
	departure_airport: one Airport,
	arrival_airport: one Airport
}{
	isBefore[departure_time, arrival_time] &&
	departure_airport != arrival_airport
}

fact {
	all f: Flight, b: Booking | f in getFlightsInBooking[b] <=> b in f.bookings // ensures that booking which uses this flight is scheduled on the flihgt and vice versa
	all f: Flight, o: Airline | f in o.flight_routes <=> o in f.operators // ensures that airline which operates a flight, has this flight in flight_routes and vice versa
	all f: Flight, a: Aircraft | f in getFlights[a] <=> a in f.aircraft // ensures that Aircraft which is used for the flight, has this flight in flights and vice versa
}

sig Passenger {
	bookings: set Booking
}

abstract sig Seat { }
sig EconomySeat extends Seat {}
sig BusinessSeat extends EconomySeat {}
sig FirstClassSeat extends BusinessSeat{}

sig Time { after: lone Time }{ isBefore[this, after]  && !isBefore[after, this] } // ensures no cycles exist in the timeline
fact { (one t: Time | Time = t.*after) } // ensures that all times have a common predecessor

pred show{some b: Booking | #b.flights > 1}
run show for 4

/*
 * Static model: Predicates
 */
// True iff t1 is strictly before t2.
pred isBefore[t1, t2: Time] { t2 in t1.^after } // Since time is totally ordered, if not in transitiv closure, first time is before second

/*
 * Static model: Functions
 */
// Returns the departure time of the given flight.
fun getDeparture[f: Flight]: Time { f.departure_time }
// Returns the arrival time of the given flight.
fun getArrival[f: Flight]: Time { f.arrival_time }
// Returns the airport the given flight departs from.
fun getOrigin[f: Flight]: Airport { f.departure_airport }
// Returns the destination airport of the given flight. 
fun getDestination[f: Flight]: Airport { f.arrival_airport }
// Returns the first flight of the given booking.
fun getFirstFlight[b: Booking]: Flight { // only flight in set that has an earlier departure than every other flight in set
	{f: b.flights | all fl: (b.flights - f) | isBefore[getDeparture[f], getDeparture[fl]]}
}
// Returns the last flight of the given booking.
fun getLastFlight[b: Booking]: Flight { // only flight in set that has an later departure than every other flight in set
	{f: b.flights | all fl: (b.flights - f) | isBefore[getDeparture[fl], getDeparture[f]]}
}
// Returns all seats of the given aircraft. 
fun getSeats[a: Aircraft]: set Seat { a.seats }
// Returns all flights for which is given aircraft is used.
fun getFlights[a: Aircraft]: set Flight { a.flights }
// Returns all bookings booked by the given passenger.
fun getBookings[p: Passenger]: set Booking { p.bookings }
// Returns all flights contained in the given booking.
fun getFlightsInBooking[b: Booking]: set Flight { b.flights } 

/*
 * Dynamic model: Functions
 */
/*
// Returns the state which comes after the given state.
fun getNextState[s: State]: State { ... } 

// Returns the location of the given passenger at the given time. 
fun getPassengerLocation[t: Time, p: Passenger]: PassengerLocation { ... }

// Returns the location of the given aircraft at the given time.
fun getAircraftLocation[t: Time, ac: Aircraft]: AircraftLocation { ... }

// Returns the time whose state the given State represents.
fun getTime[s: State]: Time { ... }
*/
