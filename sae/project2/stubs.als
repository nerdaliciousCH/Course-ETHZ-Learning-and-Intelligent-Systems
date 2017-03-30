/*
 * Static model: Signatures
 *
 * The model should contain the following (and potentially other) signatures.
 * If necessary, you have to make some of the signatures abstract and
 * make them extend other signatures.
 */

sig Aircraft extends PassengerLocation {
	seats: some Seat,
	flights: set Flight,
	location: one AircraftLocation
}{
	all disj f1, f2: flights | isBefore[getDeparture[f1], getDeparture[f2]] => isBefore[getArrival[f1], getDeparture[f2]]
	no disj f1, f2: flights | getDeparture[f1] = getDeparture[f2]
	all f: flights | getDestination[f] = getOrigin[getNextFlight[f, flights]] or no getLaterFlights[f, flights]
}

sig Airline {
	aircrafts: set Aircraft,
	flight_routes: set Flight
}

sig Airport extends AircraftLocation { }

sig Booking {
	passengers: some Passenger,
	category: one Class,
	flights: some Flight
}{
	all disj f1, f2: flights | isBefore[getDeparture[f1], getDeparture[f2]] => isBefore[getArrival[f1], getDeparture[f2]]
	all disj f1, f2: flights | getDeparture[f1] != getDeparture[f2]
}

fact {
	all f: Flight | #f.aircraft.seats >= #f.bookings.passengers
	all f: Flight | #(f.aircraft.seats & BusinessSeat) >= #(f.bookings - {b1: f.bookings | b1.category = Economy_Class}).passengers
	all f: Flight | #(f.aircraft.seats & FirstClassSeat) >= #{b1: f.bookings | b1.category = First_Class}.passengers
}

sig RoundTrip extends Booking { }{ getOrigin[getFirstFlight[this]] = getDestination[getLastFlight[this]] }

abstract sig Class { }
one sig First_Class extends Class {}
one sig Business_Class extends Class {}
one sig Economy_Class extends Class {}

sig Flight {
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
	bookings: set Booking,
	location: one PassengerLocation
}{
	// enforces that a person can only be in bookings that happen after another (no overlapping bookings)
	all disj b1, b2: {b3: Booking | this in b3.passengers} | isBefore[getDeparture[getFirstFlight[b1]], getDeparture[getFirstFlight[b2]]] => isBefore[getArrival[getLastFlight[b1]], getDeparture[getFirstFlight[b2]]]
	all disj b1, b2: {b3: Booking | this in b3.passengers} | getDeparture[getFirstFlight[b1]] != getDeparture[getFirstFlight[b2]]
}

abstract sig Seat { }
sig EconomySeat extends Seat {}
sig BusinessSeat extends EconomySeat {}
sig FirstClassSeat extends BusinessSeat{}
fact { all s: Seat | #{a: Aircraft | s in getSeats[a]} = 1 }

abstract sig Location {}
abstract sig PassengerLocation extends Location {}
abstract sig AircraftLocation extends Location {}
lone sig Unknown extends PassengerLocation {}
lone sig InAir extends AircraftLocation {}

sig Time { after: lone Time }{ isBefore[this, after]  && !isBefore[after, this] } // ensures no cycles exist in the timeline
fact { (one t: Time | Time = t.*after) } // ensures that all times have a common predecessor

/*
 * Dynamic Model
 */

sig State {}

/*
 * Predicates from Task B)
 */
pred static_instance_1 {#Flight = 1 and #Aircraft = 1 and #Airline = 1 and #Passenger = 1 and #Seat = 1 and #Airport = 2}
pred static_instance_2 { // min. for 6
	#Booking = 3 and all disj b1, b2: Booking | b1.category != b2.category
	#(Seat - BusinessSeat) = 2
	#(BusinessSeat - FirstClassSeat) = 2
	#FirstClassSeat = 2
	#Passenger = 2 and #Flight = 2 and #Airport = 2 and #Airline = 1
}
pred static_instance_3 { // Impossible because 3 flights over 2 airports implies that the first and last airports are not the same
	one r: RoundTrip | #r.flights = 3
	#Passenger = 1 and #Seat = 1 and #Airport = 2 and #Airline = 1
}
pred static_instance_4 { // min. for 6
	some disj b1, b2: Booking | #(b1.flights & b2.flights) = 1 and
			getOrigin[getFirstFlight[b1]] != getOrigin[getFirstFlight[b2]] and
			getDestination[getLastFlight[b1]] != getDestination[getLastFlight[b2]]
	all f1, f2: Flight | f1.aircraft = f2.aircraft
}
pred static_instance_5 {
	one p: Passenger | one b: p.bookings | #getFlightsInBooking[b] = 3 and
			getFirstFlight[b].aircraft = getLastFlight[b].aircraft and
			#getFlightsInBooking[b].aircraft = 2
	#Passenger = 1 and #Aircraft = 2 and #Seat = 2 and #Airline = 1
}
run static_instance_5 for 6
//run static_instance_1 for 7 but exactly 1 Flight, 1 Aircraft, 1 Airline, 1 Passenger, 1 Seat, 2 Airport

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

fun getLaterFlights[first: Flight, fs: Flight]: set Flight {
	{later: fs | isBefore[getDeparture[first], getDeparture[later]]}
}
fun getNextFlight[first: Flight, fs: Flight]: Flight{
	{n: getLaterFlights[first, fs] | all rest: (getLaterFlights[first, fs] - n) | isBefore[getDeparture[n], getDeparture[rest]]}
}

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
